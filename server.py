import os
import shutil
import warnings
from pathlib import Path

from modules import shared
from modules.block_requests import OpenMonkeyPatch, RequestBlocker
from modules.logging_colors import logger

# Load external config file from specific path
import yaml
CONFIG_PATH = Path("user_data/models/config.yaml")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {CONFIG_PATH}")
else:
    print(f"[WARNING] Config file not found at {CONFIG_PATH}")

# Set up Gradio temp directory path
gradio_temp_path = Path('user_data') / 'cache' / 'gradio'
shutil.rmtree(gradio_temp_path, ignore_errors=True)
gradio_temp_path.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ.update({
    'GRADIO_ANALYTICS_ENABLED': 'False',
    'BITSANDBYTES_NOWELCOME': '1',
    'GRADIO_TEMP_DIR': str(gradio_temp_path)
})

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='The value passed into gr.Dropdown()')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_names" has conflict')

with RequestBlocker():
    from modules import gradio_hijack
    import gradio as gr

import matplotlib
matplotlib.use('Agg')

import json
import signal
import sys
import time
from functools import partial
from threading import Lock, Thread

import modules.extensions as extensions_module
from modules import (
    training,
    ui,
    ui_chat,
    ui_default,
    ui_file_saving,
    ui_model_menu,
    ui_notebook,
    ui_parameters,
    ui_session,
    utils
)
from modules.chat import generate_pfp_cache
from modules.extensions import apply_extensions
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model_if_idle
from modules.models_settings import (
    get_fallback_settings,
    get_model_metadata,
    update_gpu_layers_and_vram,
    update_model_parameters
)
from modules.shared import do_cmd_flags_warnings
from modules.utils import gradio

def signal_handler(sig, frame):
    logger.info("Received Ctrl+C. Shutting down Text generation web UI gracefully.")
    if shared.model and shared.model.__class__.__name__ == 'LlamaServer':
        try:
            shared.model.stop()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def create_interface():
    title = 'Text generation web UI'
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    if shared.args.extensions:
        extensions_module.load_extensions()

    shared.persistent_interface_state.update({
        'mode': shared.settings['mode'],
        'loader': shared.args.loader or 'llama.cpp',
        'filter_by_loader': (shared.args.loader or 'All') if not shared.args.portable else 'llama.cpp'
    })

    for cache_file in ['pfp_character.png', 'pfp_character_thumb.png']:
        cache_path = Path(f"user_data/cache/{cache_file}")
        if cache_path.exists():
            cache_path.unlink()

    if shared.settings['mode'] != 'instruct':
        generate_pfp_cache(shared.settings['character'])

    css = ui.css + apply_extensions('css')
    js = ui.js + apply_extensions('js')

    shared.input_elements = ui.list_interface_input_elements()

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        if Path("user_data/notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="user_data/notification.mp3", elem_id="audio_notification", visible=False)

        ui_file_saving.create_ui()
        shared.gradio['temporary_text'] = gr.Textbox(visible=False)

        ui_chat.create_ui()
        ui_default.create_ui()
        ui_notebook.create_ui()
        ui_parameters.create_ui()
        ui_model_menu.create_ui()
        if not shared.args.portable:
            training.create_ui()
        ui_session.create_ui()

        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()
        ui_model_menu.create_event_handlers()

        ui.setup_auto_save()

        shared.gradio['interface'].load(
            None,
            gradio('show_controls'),
            None,
            js=f"""(x) => {{
                const savedTheme = localStorage.getItem('theme');
                const serverTheme = {str(shared.settings['dark_theme']).lower()} ? 'dark' : 'light';
                if (!savedTheme || !sessionStorage.getItem('theme_synced')) {{
                    localStorage.setItem('theme', serverTheme);
                    sessionStorage.setItem('theme_synced', 'true');
                    if (serverTheme === 'dark') {{
                        document.getElementsByTagName('body')[0].classList.add('dark');
                    }} else {{
                        document.getElementsByTagName('body')[0].classList.remove('dark');
                    }}
                }} else {{
                    if (savedTheme === 'dark') {{
                        document.getElementsByTagName('body')[0].classList.add('dark');
                    }} else {{
                        document.getElementsByTagName('body')[0].classList.remove('dark');
                    }}
                }}
                {js}
                {ui.show_controls_js}
                toggle_controls(x);
            }}"""
        )

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, gradio(ui.list_interface_input_elements()), show_progress=False)

        extensions_module.create_extensions_tabs()
        extensions_module.create_extensions_block()

    shared.gradio['interface'].launch(
        max_threads=64,
        prevent_thread_lock=True,
        share=shared.args.share,
        server_name='0.0.0.0',
        server_port=shared.args.listen_port,
        inbrowser=shared.args.auto_launch,
        auth=auth or None,
        ssl_verify=False if (shared.args.ssl_keyfile or shared.args.ssl_certfile) else True,
        ssl_keyfile=shared.args.ssl_keyfile,
        ssl_certfile=shared.args.ssl_certfile,
        root_path=shared.args.subpath,
        allowed_paths=["css", "js", "extensions", "user_data/cache"]
    )

if __name__ == "__main__":
    logger.info("Starting Text generation web UI")
    do_cmd_flags_warnings()

    settings_file = None
    if shared.args.settings and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('user_data/settings.yaml').exists():
        settings_file = Path('user_data/settings.yaml')
    elif Path('user_data/settings.json').exists():
        settings_file = Path('user_data/settings.json')

    if settings_file:
        logger.info(f"Loading settings from \"{settings_file}\"")
        with open(settings_file, 'r', encoding='utf-8') as f:
            contents = f.read()
            new_settings = json.loads(contents) if settings_file.suffix == ".json" else yaml.safe_load(contents)
            shared.settings.update(new_settings)

    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)

    extensions_module.available_extensions = utils.get_available_extensions()
    available_models = utils.get_available_models()

    if shared.args.model:
        shared.model_name = shared.args.model
    elif shared.args.model_menu:
        if not available_models:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')
            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()
        shared.model_name = available_models[i]

    if shared.model_name != 'None':
        p = Path(shared.model_name)
        if p.exists():
            model_name = p.parts[-1]
            shared.model_name = model_name
        else:
            model_name = shared.model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings, initial=True)

        if 'gpu_layers' not in shared.provided_arguments and shared.args.loader == 'llama.cpp' and 'gpu_layers' in model_settings:
            _, adjusted_layers = update_gpu_layers_and_vram(
                shared.args.loader,
                model_name,
                model_settings['gpu_layers'],
                shared.args.ctx_size,
                shared.args.cache_type,
                auto_adjust=True,
                for_ui=False
            )
            shared.args.gpu_layers = adjusted_layers

        shared.model, shared.tokenizer = load_model(model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    shared.generation_lock = Lock()

    if shared.args.idle_timeout > 0:
        timer_thread = Thread(target=unload_model_if_idle)
        timer_thread.daemon = True
        timer_thread.start()

    if shared.args.nowebui:
        shared.args.extensions = [x for x in (shared.args.extensions or []) if x != 'gallery']
        if shared.args.extensions:
            extensions_module.load_extensions()
    else:
        create_interface()
        while True:
            time.sleep(0.5)
            if shared.need_restart:
                shared.need_restart = False
                time.sleep(0.5)
                shared.gradio['interface'].close()
                time.sleep(0.5)
                create_interface()
