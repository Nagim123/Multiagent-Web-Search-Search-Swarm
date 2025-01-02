import re
import random
import string
import requests
import time
from bs4 import BeautifulSoup
from bs4.element import Comment

import gym

END_BUTTON = 'Buy Now'

def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg

class WebAgentSiteEnv(gym.Env):
    def __init__(self, observation_mode: str = 'text'):
        super(WebAgentSiteEnv, self).__init__()
        self.session = None
        self.core_url = ""
        self.page_name = ""
        self.current_product = ""
        self.search_keywords = []
        self.selected_attrs = {}
        self.available_attrs = {}
        self.reset()

    def get_reward(self):
        """Get reward value at current step of the environment"""
        html_obj = self._parse_html(self.current_url)
        r = html_obj.find(id='reward')
        r = float(r.findChildren("pre")[0].string) if r is not None else 0.0
        return r
    
    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        texts = self._parse_html(self.current_url).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        observation = ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        return observation


    def step(self, action: str):
        reward = 0.0
        done = False
        info = None

        # Map action to executed command on the WebShop environment via the broswer driver
        action_name, action_arg = parse_action(action)
        if action_name == 'search':
            self.search_keywords = str(action_arg.split(" ")).lower()
            self.page_name = "search_results"
            self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.search_keywords}/1"
        elif action_name == 'click':
            if action_arg == "Back to Seach":
                self.search_keywords = []
                self.page_name = ""
                self.current_url = f"{self.core_url}/{self.session}"
            elif action_arg == "< Prev":
                if self.page_name == "item_page":
                    self.current_product = ""
                    self.page_name = "search_results"
                    self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.search_keywords}/1"
                elif self.page_name == "item_sub_page":
                    self.page_name = "item_page"
                    self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/{self.selected_attrs}"
            elif action_arg == "Description":
                self.page_name = "item_sub_page"
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/Description/{self.selected_attrs}"
            elif action_arg == "Features":
                self.page_name = "item_sub_page"
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/Features/{self.selected_attrs}"
            elif action_arg == "Reviews":
                self.page_name = "item_sub_page"
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/Reviews/{self.selected_attrs}"
            elif action_arg == "Attributes":
                self.page_name = "item_sub_page"
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/Attributes/{self.selected_attrs}"
            elif action_arg == "Buy Now":
                self.page_name = "done"
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.selected_attrs}"
                done = True
                reward = self.get_reward()
            elif self.page_name == "item_page":
                for option in self.available_attrs.keys():
                    if action_arg in self.available_attrs[option]:
                        self.selected_attrs[option] = action_arg
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/{self.selected_attrs}"
            elif self.page_name == "search_results":
                self.page_name = "item_page"
                self.current_product = action_arg
                self.current_url = f"{self.core_url}/{self.page_name}/{self.session}/{self.current_product}/{self.search_keywords}/1/{self.selected_attrs}"
        elif action_name == 'end':
            done = True
        else:
            print('Invalid action. No action performed.')
        return self.observation, reward, done, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        # Determine if a search bar is available
        html_obj = self._parse_html(self.current_url)
        has_search_bar = not (html_obj.find("input", {"id": "search_input"}) is None)

        # Collect buttons, links, and options as clickables
        buttons = [el.get_text() for el in html_obj.find_all(class_="btn")]
        product_links = [el.get_text() for el in html_obj.find_all(class_="product-link")]
        buying_options = []
        self.available_attrs = {}
        for el in html_obj.find_all("input", {"type": "radio"}):
            if el.get("name") not in self.available_attrs:
                self.available_attrs[el.get("name")] = [el.get("value")]
            else:
                self.available_attrs[el.get("name")].append(el.get("value"))
            buying_options.append(el.get("value"))
        return dict(
            has_search_bar=has_search_bar,
            clickables=buttons+product_links+buying_options,
        )


    def _parse_html(self, url: str):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        html = requests.get(url)
        html_obj = BeautifulSoup(html.text, 'html.parser')
        return html_obj
    
    def get_instruction_text(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.current_url)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text
    
    

    def reset(self, idx=None):
        """Create a new session and reset environment variables"""
        self.page_name = ""
        self.current_product = ""
        self.search_keywords = []
        self.selected_attrs = {}
        self.available_attrs = {}
        
        if idx is None:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        else:
            self.session = f"fixed_{idx}"
        self.core_url = f'http://localhost:3000'
        self.current_url = f"{self.core_url}/{self.session}"
        self.instruction_text = self.get_instruction_text()

        return self.observation, None
    
    def close(self):
        pass

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )