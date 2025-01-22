# Running other methods
Every methods is already adapted to use LLama-3-8b from HuggingFace! (You need an [access](https://huggingface.co/meta-llama/Meta-Llama-3-8B) to this model on HuggingFace provided by Meta)

* **ReAct**: Use Jupyter notebook
* **LATS**: follow [README](https://github.com/lapisrocks/LanguageAgentTreeSearch) and run command
```terminal
python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/new_run.log \
```
* **Reflexion**: follow [README](https://github.com/noahshinn/reflexion) and run command
```terminal
python main.py \
        --num_trials 10 \
        --num_envs 100 \
        --run_name "reflexion_run_logs" \
```
* **ADaPT**: follow [README](https://github.com/archiki/ADaPT) and run command
```terminal
python run_webshop.py
```