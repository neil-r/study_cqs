from typing import List

from driver_00_setup import datasets, event_types, p_factories

from study_llm.event_extraction_task.dataset_maven import read_in_dataset, MavenParagraph


for dataset_file_path in datasets:
    extraction_tasks:List[MavenParagraph] = read_in_dataset(
        dataset_file_path,
        event_types=event_types
    )

    print(f"The number of tasks is {len(list(x for x in extraction_tasks if len(x.events) > 0))}")

    for prompt_factory in p_factories:

        for task in extraction_tasks:
            prompt_handler = prompt_factory.generate_prompt(task)
            print("\n\n-----Example Prompt-----\n")
            print(prompt_handler.content)

            print("\n-----Event mentions of interest in content-----\n")
            print(task.events)

            result = input("Another (y = for yes)?")

            if result != "y":
                break
