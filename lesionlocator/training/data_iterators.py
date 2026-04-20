import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager

from time import sleep
from typing import List
import torch

from lesionlocator.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from lesionlocator.utilities.prompt_handling.prompt_handler import get_prompt_from_inst_or_bin_seg, get_prompt_from_json

import numpy as np

def preprocess_fromfiles_save_to_queue(input_files: List[str],
                                       prompt_files: List[str],
                                       output_files: List[str],
                                       prompt_type: str,
                                       plans_config: dict,
                                       #plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_config: str, 
                                       modality: str,
                                       #configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False,
                                       track: bool = False,
                                       train: bool = False):
    # print("Starting preprocessing worker",flush=True)
    plans_manager = PlansManager(plans_config)
    configuration_manager = plans_manager.get_configuration(configuration_config, modality=modality)
    configuration_manager.set_preprocessor_name('TrainingPreprocessor')

    # print(configuration_config, modality, flush=True)

    try:
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(input_files)):
            # input_files[idx] is either a str (single-channel) or List[str] (multi-channel).
            # The preprocessor's run_case always expects List[str].
            _image_files = input_files[idx] if isinstance(input_files[idx], list) else [input_files[idx]]

            if prompt_files[idx].endswith('.json'):
                # print(f'Using JSON prompts for case {input_files[idx]}', flush=True)
                seg = None
                data, _, data_properties, bl_data, bl_data_properties = preprocessor.run_case(_image_files,
                                                                                                None,
                                                                                                plans_manager,
                                                                                                configuration_manager,
                                                                                                dataset_json,
                                                                                                track)


                prompt = get_prompt_from_json(prompt_files[idx], prompt_type, data_properties, data.shape[1:])
                # print(f'finished preprocessing for {input_files[idx]}', flush=True)
            else:
                # print(f'Using segmentation prompts for case {input_files[idx]}', flush=True)
                data, seg, data_properties, bl_data, bl_data_properties = preprocessor.run_case(_image_files,
                                                                    prompt_files[idx],
                                                                    plans_manager,
                                                                    configuration_manager,
                                                                    dataset_json,
                                                                    track, train=train)
                prompt = get_prompt_from_inst_or_bin_seg(seg, prompt_type)
                # print(f'finished preprocessing for {input_files[idx]}', flush=True)
            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            item = {'data': data, 'prompt': prompt, 'seg':seg, 'data_properties': data_properties, 'ofile': output_files[idx], 'bl_data': bl_data, 'bl_data_properties': bl_data_properties}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(input_files: List[str],
                                     prompt_files: List[str],
                                     output_files: List[str],
                                     prompt_type: str,
                                     # plans_manager: PlansManager,
                                     plans_config: dict,
                                     dataset_json: dict,
                                     configuration_config: str,
                                     modality: str,
                                     # configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False,
                                     track: bool = False,
                                     train: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(input_files), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        # print(f'Starting background worker {i}, processing {len(input_files[i::num_processes])} of {len(input_files)} cases')
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         input_files[i::num_processes],
                         prompt_files[i::num_processes],
                         output_files[i::num_processes],
                         prompt_type,
                         plans_config,
                         dataset_json,
                         configuration_config,
                         modality,
                         queue,
                         event,
                         abort_event,
                         verbose,
                         track,
                         train
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)
    
    while True:
        found = False
        for i in range(num_processes):
            if not target_queues[i].empty():
                item = target_queues[i].get()
                found = True
                if pin_memory:
                    [v.pin_memory() for v in item.values() if isinstance(v, torch.Tensor)]
                yield item
        if not found:
            all_done = all([done_events[i].is_set() and target_queues[i].empty() for i in range(num_processes)])
            all_ok = all([p.is_alive() or done_events[i].is_set() for i, p in enumerate(processes)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                'workers or get more RAM in that case!')
            if all_done:
                break
            sleep(0.01)
    [ p.join() for p in processes ]

    # worker_ctr = 0
    # while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
    #     # import IPython;IPython.embed()
    #     if not target_queues[worker_ctr].empty():
    #         item = target_queues[worker_ctr].get()
    #         worker_ctr = (worker_ctr + 1) % num_processes
    #     else:
    #         all_ok = all(
    #             [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
    #         if not all_ok:
    #             raise RuntimeError('Background workers died. Look for the error message further up! If there is '
    #                                'none then your RAM was full and the worker was killed by the OS. Use fewer '
    #                                'workers or get more RAM in that case!')
    #         sleep(0.01)
    #         continue
    #     if pin_memory:
    #         [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
    #     yield item
    # [p.join() for p in processes]
