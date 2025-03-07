import study_llm.db as db
from study_llm.hugging_face import get_model_info

from driver_00_setup import huggingface_models


# Prepare the database that will store the discussion results
database = db.DatabaseSqlLite()


if __name__ == "__main__": 
    for model_id in huggingface_models:
        # Retrieve Model Info
        model1 = get_model_info(model_id)

        # Unpacking dictionary values into separate variables
        model1_id, model1_name, model1_size, model1_version = (
            model1["model_id"],
            model1["model_name"],
            model1["model_size"],
            model1["model_version"],
        )
        database.add_model_entry(model1_id, model1_name, model1_size, model1_version)

    # save models
    #database.add_model_entry("simpleA", "Testing Model", 1, 1.0)
    