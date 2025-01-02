import gradio as gr
import json, time
from tqdm import tqdm

from config_reader import ConfigReader
from agents.search_swarm_amazon import SearchSwarm
from agents.test_primitive_agent import PrimitiveAgent
from amazon_test.webshop_lite import dict_to_fake_html
from amazon_test.predict_help import (
    Page, convert_dict_to_actions, convert_html_to_text,
    parse_results_amz, parse_item_page_amz,
    parse_results_ws, parse_item_page_ws,
    parse_results_ebay, parse_item_page_ebay,
    WEBSHOP_URL, WEBSHOP_SESSION
)

ConfigReader("config.json")

ENVIRONMENTS = ['amazon', 'webshop', 'ebay']

def process_str(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s


def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

def get_return_value(env, asin, options, search_terms, page_num, product):
    asin_url = None

    # Determine product URL + options based on environment
    if env == 'webshop':
        query_str = "+".join(search_terms.split())
        options_str = json.dumps(options)
        asin_url = (
            f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/'
            f'{asin}/{query_str}/{page_num}/{options_str}'
        )
    else:
        asin_url = f"https://www.ebay.com/itm/{asin}" if env == 'ebay' else \
            f"https://www.amazon.com/dp/{asin}"
    
    # Extract relevant fields for product
    product_reduced = {k: v for k, v in product.items() if k in ["asin", "Title", "Description", "BulletPoints"]}
    product_reduced["Description"] = product_reduced["Description"][:100] + "..."
    product_reduced["Features"] = product_reduced.pop("BulletPoints")
    product_reduced["Features"] = "\n".join(product_reduced["Features"][:100]) + "..."

    # Create HTML to show link to product
    html = """<!DOCTYPE html><html><head><title>Chosen Product</title></head><body>"""
    html += f"""Product Image:<img src="{product["MainImage"]}" height="50px" /><br>""" if len(product["MainImage"]) > 0 else ""
    html += f"""Link to Product:
        <a href="{asin_url}" style="color:blue;text-decoration:underline;" target="_blank">{asin_url}</a>
        </body></html>"""

    return product_reduced, options if len(options) > 0 else {"options": "None Selected"}, html

def run_episode(goal, env, verbose=True):
    """
    Interact with amazon to find a product given input goal.
    Input: text goal
    Output: a url of found item on amazon.
    """
    agent = SearchSwarm()
    env = env.lower()
    if env not in ENVIRONMENTS:
        print(f"[ERROR] Environment {env} not recognized")
        
    obs = "Amazon Shopping Game\n [SEP] Instruction: [SEP] " + goal + " [SEP] \n[button] [SEP] search [button]"
    info = {'valid': ['search[stuff]']}
    product_map = {}
    title_to_asin_map = {}
    search_results_cache = {}
    visited_asins, clicked_options = set(), set()
    sub_page_type, page_type, page_num = None, None, None
    search_terms, prod_title, asin = None, None, None
    options = {}
    
    for i in range(216):
        # Run prediction
        clickables = [v[v.find('[')+1:v.rfind(']')].strip() if v[:v.find('[')] != "search" else "search" for v in info["valid"]]
        action = agent.act(obs, {"clickables": clickables})
        if verbose:
            print("====")
            print(action)
        
        # Previous Page Type, Action -> Next Page Type
        action_content = action[action.find("[")+1:action.rfind("]")]
        prev_page_type = page_type
        if action.startswith('search['):
            page_type = Page.RESULTS
            search_terms = action_content
            page_num = 1
        elif action.startswith('click['):
            if action.startswith('click[item -'):
                prod_title = action_content[len("item -"):].strip()
                found = False
                for key in title_to_asin_map:
                    if prod_title == key:
                        asin = title_to_asin_map[key]
                        page_type = Page.ITEM_PAGE
                        visited_asins.add(asin)
                        found = True
                        break
                if not found:
                    raise Exception("Product to click not found")
                    
            elif any(x.value in action for x in [Page.DESC, Page.FEATURES, Page.REVIEWS]):
                page_type = Page.SUB_PAGE
                sub_page_type = Page(action_content.lower())
                
            elif action == 'click[< prev]':
                if sub_page_type is not None:
                    page_type, sub_page_type = Page.ITEM_PAGE, None
                elif prev_page_type == Page.ITEM_PAGE:
                    page_type = Page.RESULTS
                    options, clicked_options = {}, set()
                elif prev_page_type == Page.RESULTS and page_num > 1:
                    page_type = Page.RESULTS
                    page_num -= 1
                    
            elif action == 'click[next >]':
                page_type = Page.RESULTS
                page_num += 1
                
            elif action.lower() == 'click[back to search]':
                page_type = Page.SEARCH
                
            elif action == 'click[buy now]':
                return get_return_value(env, asin, options, search_terms, page_num, product_map[asin])
            
            elif prev_page_type == Page.ITEM_PAGE:
                found = False
                for opt_name, opt_values in product_map[asin]["options"].items():
                    strip_opt_values = [opt.strip() for opt in opt_values]
                    if action_content in strip_opt_values:
                        options[opt_name] = action_content
                        page_type = Page.ITEM_PAGE
                        clicked_options.add(action_content)
                        found = True
                        break
                if not found:
                    raise Exception("Unrecognized action: " + action)
        else:
            raise Exception("Unrecognized action:" + action)
        
        if verbose:
            print(f"Parsing {page_type.value} page...")
        
        # URL -> Real HTML -> Dict of Info
        if page_type == Page.RESULTS:
            if search_terms in search_results_cache:
                data = search_results_cache[search_terms]
                if verbose:
                    print(f"Loading cached results page for \"{search_terms}\"")
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_results_amz(search_terms, page_num, verbose)
                if env == 'webshop':
                    data = parse_results_ws(search_terms, page_num, verbose)
                if env == 'ebay':
                    data = parse_results_ebay(search_terms, page_num, verbose)
                end = time.time()
                if verbose:
                    print(f"Parsing search results took {end-begin} seconds")

                search_results_cache[search_terms] = data
                for d in data:
                    title_to_asin_map[d['Title']] = d['asin']
        elif page_type == Page.ITEM_PAGE or page_type == Page.SUB_PAGE:
            if asin in product_map:
                if verbose:
                    print("Loading cached item page for", asin)
                data = product_map[asin]
            else:
                begin = time.time()
                if env == 'amazon':
                    data = parse_item_page_amz(asin, verbose)
                if env == 'webshop':
                    data = parse_item_page_ws(asin, search_terms, page_num, options, verbose)
                if env == 'ebay':
                    data = parse_item_page_ebay(asin, verbose)
                end = time.time()
                if verbose:
                    print("Parsing item page took", end-begin, "seconds")
                product_map[asin] = data
        elif page_type == Page.SEARCH:
            if verbose:
                print("Executing search")
            obs = "Amazon Shopping Game\nInstruction:" + goal + "\n[button] search [button]"
            info = {'valid': ['search[stuff]']}
            continue
        else:
            raise Exception("Page of type `", page_type, "` not found")

        # Dict of Info -> Fake HTML -> Text Observation
        begin = time.time()
        html_str = dict_to_fake_html(data, page_type, asin, sub_page_type, options, product_map, goal)
        obs = convert_html_to_text(html_str, simple=True, clicked_options=clicked_options, visited_asins=visited_asins)
        end = time.time()
        if verbose:
            print("[Page Info -> WebShop HTML -> Observation] took", end-begin, "seconds")

        # Dict of Info -> Valid Action State (Info)
        begin = time.time()
        prod_arg = product_map if page_type == Page.ITEM_PAGE else data
        info = convert_dict_to_actions(page_type, prod_arg, asin, page_num)
        end = time.time()
        if verbose:
            print("Extracting available actions took", end-begin, "seconds")

mode = input("Choose a mode, (I) interface, (T) test the Amazon json file: ")

if mode == "I":
    gr.Interface(
        fn=run_episode,
        inputs=[
            gr.Textbox(lines=7, label="Input Text"),
            gr.Radio(['Amazon', 'eBay'], type="value", label='Environment')
        ],
        outputs=[
            gr.JSON(label="Selected Product"),
            gr.JSON(label="Selected Options"),
            gr.HTML()
        ],
        examples=[
            ["I want to find a gold floor lamp with a glass shade and a nickel finish that i can use for my living room, and price lower than 270.00 dollars", "Amazon"],
            ["I need some cute heart-shaped glittery cupcake picks as a gift to bring to a baby shower", "Amazon"],
            ["I want to buy ballet shoes which have rubber sole in grey suede color and a size of 6", "Amazon"],
            ["I would like a 7 piece king comforter set decorated with flowers and is machine washable", "Amazon"],
            ["I'm trying to find white bluetooth speakers that are not only water resistant but also come with stereo sound", "eBay"],
            ["find me the soy free 3.5 ounce 4-pack of dang thai rice chips, and make sure they are the aged cheddar flavor.  i also need the ones in the resealable bags", "eBay"],
            ["I am looking for a milk chocolate of 1 pound size in a single pack for valentine day", "eBay"],
            ["I'm looking for a mini pc intel core desktop computer which supports with windows 11", "eBay"]
        ],
        title="WebShop",
        article="<p style='padding-top:15px;text-align:center;'>To learn more about this project, check out the <a href='https://webshop-pnlp.github.io/' target='_blank'>project page</a>!</p>",
        description="<p style='text-align:center;'>Sim-to-real transfer of agent trained on WebShop to search a desired product on Amazon from any natural language query!</p>",
    ).launch(inline=False)
elif mode == "T":
    products_json = []
    with open("./other/product_list.json", "r") as product_file:
        products_json = json.load(product_file)
    
    results_data = []
    with open("./other/output.json", "r") as output_file:
        results_data = json.load(output_file)
    
    progress = tqdm(products_json[len(results_data):])
    for item in progress:
        result = run_episode(item["query"], "Amazon",verbose=False)
        if result is None:
            raise Exception("No product selected by model")
        output_product, options, html = result
        results_data.append({"output_item": f"https://www.amazon.com/dp/{output_product['asin']}/", "true_item": item["link"], "options": options})
        with open("./other/output.json", "w") as output_file:
            output_file.write(json.dumps(results_data))