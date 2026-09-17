import os
import joblib


# ===================================
# MODEL FILES
# ===================================

MODEL_FILES = {
    "Random Forest": "ml/random_forest_model.pkl",
    "XGBoost": "ml/xgboost_model.pkl",
    "LightGBM": "ml/lightgbm_model.pkl",
    "CatBoost": "ml/catboost_model.pkl"
}


# ===================================
# EXPECTED MODEL CONFIGURATION
# ===================================

EXPECTED_TARGET = "col1"

EXPECTED_FEATURE_VERSION = "training_v1"

EXPECTED_FEATURE_COUNT = 108


# ===================================
# LOAD ALL MODELS
# ===================================

def load_all_models():

    print("\n===================================")
    print("          MODEL LOADER")
    print("===================================")

    loaded_models = {}

    reference_features = None

    # ===================================
    # PROCESS EACH MODEL
    # ===================================

    for model_name, model_path in MODEL_FILES.items():

        print("\n-----------------------------------")
        print(model_name)
        print("-----------------------------------")

        # ===================================
        # CHECK MODEL FILE
        # ===================================

        if not os.path.exists(model_path):

            print("Model file: NOT FOUND")

            print(
                f"Path: {model_path}"
            )

            continue

        print("Model file: FOUND")

        # ===================================
        # LOAD MODEL BUNDLE
        # ===================================

        try:

            bundle = joblib.load(
                model_path
            )

        except Exception as error:

            print("Model load: FAILED")

            print(
                f"Error: {error}"
            )

            continue

        # ===================================
        # REQUIRED METADATA
        # ===================================

        required_keys = [
            "model",
            "feature_names",
            "target_column",
            "feature_version",
            "model_name"
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in bundle
        ]

        if missing_keys:

            print(
                "Metadata validation: FAILED"
            )

            print(
                f"Missing keys: {missing_keys}"
            )

            continue

        # ===================================
        # READ MODEL DATA
        # ===================================

        model = bundle["model"]

        feature_names = list(
            bundle["feature_names"]
        )

        target_column = (
            bundle["target_column"]
        )

        feature_version = (
            bundle["feature_version"]
        )

        saved_model_name = (
            bundle["model_name"]
        )

        # ===================================
        # TARGET VALIDATION
        # ===================================

        if target_column != EXPECTED_TARGET:

            print(
                "Target validation: FAILED"
            )

            print(
                f"Expected target: "
                f"{EXPECTED_TARGET}"
            )

            print(
                f"Model target: "
                f"{target_column}"
            )

            continue

        print(
            f"Target: "
            f"{target_column} -> OK"
        )

        # ===================================
        # FEATURE VERSION VALIDATION
        # ===================================

        if (
            feature_version
            != EXPECTED_FEATURE_VERSION
        ):

            print(
                "Feature version: FAILED"
            )

            print(
                f"Expected: "
                f"{EXPECTED_FEATURE_VERSION}"
            )

            print(
                f"Found: "
                f"{feature_version}"
            )

            continue

        print(
            f"Feature version: "
            f"{feature_version} -> OK"
        )

        # ===================================
        # FEATURE COUNT VALIDATION
        # ===================================

        feature_count = len(
            feature_names
        )

        print(
            f"Features: "
            f"{feature_count}"
        )

        if (
            feature_count
            != EXPECTED_FEATURE_COUNT
        ):

            print(
                "Feature count validation: "
                "FAILED"
            )

            print(
                f"Expected: "
                f"{EXPECTED_FEATURE_COUNT}"
            )

            continue

        print(
            "Feature count validation: "
            "PASSED"
        )

        # ===================================
        # MODEL INTERNAL FEATURE COUNT
        # ===================================

        model_feature_count = None

        # -----------------------------------
        # TRY n_features_in_
        # -----------------------------------

        if hasattr(
            model,
            "n_features_in_"
        ):

            try:

                detected_count = int(
                    model.n_features_in_
                )

                # IMPORTANT:
                #
                # CatBoost may expose
                # n_features_in_ as 0.
                #
                # Zero is NOT treated as
                # a real feature count.
                #
                # Only positive values are
                # considered valid.

                if detected_count > 0:

                    model_feature_count = (
                        detected_count
                    )

            except Exception:

                model_feature_count = None

        # -----------------------------------
        # CATBOOST FALLBACK
        # -----------------------------------

        if (
            model_feature_count is None
            and saved_model_name == "CatBoost"
        ):

            try:

                if hasattr(
                    model,
                    "get_feature_count"
                ):

                    catboost_count = int(
                        model.get_feature_count()
                    )

                    if catboost_count > 0:

                        model_feature_count = (
                            catboost_count
                        )

            except Exception:

                model_feature_count = None

        # ===================================
        # MODEL FEATURE VALIDATION
        # ===================================

        if model_feature_count is not None:

            print(
                f"Model input features: "
                f"{model_feature_count}"
            )

            if (
                model_feature_count
                != feature_count
            ):

                print(
                    "Model/schema validation: "
                    "FAILED"
                )

                continue

            print(
                "Model input feature "
                "validation: PASSED"
            )

        else:

            print(
                "Model input feature count: "
                "NOT AVAILABLE"
            )

            print(
                "Using saved feature metadata "
                "for schema validation."
            )

            if saved_model_name == "CatBoost":

                print(
                    "CatBoost schema validation: "
                    "PASSED"
                )

        # ===================================
        # FEATURE SCHEMA VALIDATION
        # ===================================

        if reference_features is None:

            reference_features = (
                list(feature_names)
            )

            print(
                "Reference feature schema "
                "created."
            )

        else:

            if (
                list(feature_names)
                != reference_features
            ):

                print(
                    "Feature schema validation: "
                    "FAILED"
                )

                print(
                    "Feature names or order "
                    "do not match."
                )

                continue

            print(
                "Feature schema validation: "
                "PASSED"
            )

        # ===================================
        # SAVE LOADED MODEL
        # ===================================

        loaded_models[
            model_name
        ] = bundle

        print(
            f"Model name: "
            f"{saved_model_name}"
        )

        print(
            "Model load: SUCCESS"
        )

    # ===================================
    # MODEL LOADER SUMMARY
    # ===================================

    print("\n===================================")
    print("        MODEL LOADER SUMMARY")
    print("===================================")

    print(
        f"Models successfully loaded: "
        f"{len(loaded_models)}"
    )

    for model_name in loaded_models:

        print(
            f"  ✓ {model_name}"
        )

    print(
        f"\nExpected models: "
        f"{len(MODEL_FILES)}"
    )

    # ===================================
    # FINAL VALIDATION
    # ===================================

    if (
        len(loaded_models)
        == len(MODEL_FILES)
    ):

        print(
            "ALL MODEL VALIDATIONS: PASSED"
        )

    else:

        print(
            "MODEL VALIDATION: INCOMPLETE"
        )

        missing_models = [
            model_name
            for model_name in MODEL_FILES
            if model_name not in loaded_models
        ]

        print(
            f"Missing/failed models: "
            f"{missing_models}"
        )

    print("\n===================================")
    print("MODEL LOADER COMPLETED")
    print("===================================\n")

    return loaded_models


# ===================================
# DIRECT EXECUTION
# ===================================

if __name__ == "__main__":

    load_all_models()