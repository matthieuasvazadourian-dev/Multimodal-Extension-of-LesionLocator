import multiprocessing
import os
import queue
import time
import traceback
import torch.multiprocessing as torch_mp
from torch.multiprocessing import Event, Queue

try:
    torch_mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

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
    worker_pid = os.getpid()
    print(f'[worker pid={worker_pid}] started, {len(input_files)} cases assigned', flush=True)
    plans_manager = PlansManager(plans_config)
    configuration_manager = plans_manager.get_configuration(configuration_config, modality=modality)
    configuration_manager.set_preprocessor_name('TrainingPreprocessor')

    try:
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(input_files)):
            case_start = time.time()
            case_label = input_files[idx][0] if isinstance(input_files[idx], list) else input_files[idx]
            print(f'[worker pid={worker_pid}] case {idx+1}/{len(input_files)} start: {os.path.basename(str(case_label))}', flush=True)
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
            else:
                data, seg, data_properties, bl_data, bl_data_properties = preprocessor.run_case(_image_files,
                                                                    prompt_files[idx],
                                                                    plans_manager,
                                                                    configuration_manager,
                                                                    dataset_json,
                                                                    track, train=train)
                prompt = get_prompt_from_inst_or_bin_seg(seg, prompt_type)
            case_elapsed = time.time() - case_start
            print(f'[worker pid={worker_pid}] case {idx+1}/{len(input_files)} done in {case_elapsed:.1f}s', flush=True)
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
        print(f'[worker pid={worker_pid}] all {len(input_files)} cases done', flush=True)
    except Exception as e:
        print(f'[worker pid={worker_pid}] CRASHED: {e}\n{traceback.format_exc()}', flush=True)
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
    num_processes = min(len(input_files), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = context.Event()
    print(f'[iterator] spawning {num_processes} preprocessing workers for {len(input_files)} cases', flush=True)
    for i in range(num_processes):
        event = context.Event()
        queue = context.Queue(maxsize=2)
        n_assigned = len(input_files[i::num_processes])
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
        print(f'[iterator] worker {i} started (pid={pr.pid}), {n_assigned} cases', flush=True)
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    items_per_worker = [len(input_files[i::num_processes]) for i in range(num_processes)]
    items_received = [0] * num_processes
    total_expected = sum(items_per_worker)
    items_yielded = 0
    last_heartbeat = time.time()

    try:
        while items_yielded < total_expected:
            found = False
            for i in range(num_processes):
                if items_received[i] < items_per_worker[i]:
                    try:
                        item = target_queues[i].get(timeout=0.01)
                    except queue.Empty:
                        continue
                    items_received[i] += 1
                    found = True
                    items_yielded += 1
                    last_heartbeat = time.time()
                    if pin_memory:
                        item = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
                    yield item
            if not found:
                all_ok = all(
                    items_received[i] >= items_per_worker[i] or p.is_alive()
                    for i, p in enumerate(processes)
                ) and not abort_event.is_set()
                if not all_ok:
                    raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                       'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                       'workers or get more RAM in that case!')
                if time.time() - last_heartbeat > 60:
                    statuses = ', '.join(
                        f'w{i}:{"alive" if processes[i].is_alive() else "dead"}'
                        for i in range(num_processes)
                    )
                    print(f'[iterator] waiting for workers ({statuses}), {items_yielded}/{total_expected} cases yielded so far', flush=True)
                    last_heartbeat = time.time()
                sleep(0.01)
        print(f'[iterator] done, {items_yielded} cases yielded total', flush=True)
    finally:
        abort_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        for q in target_queues:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
