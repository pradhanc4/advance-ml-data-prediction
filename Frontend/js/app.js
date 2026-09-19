const API_BASE_URL = "http://127.0.0.1:5000";

const runPredictionBtn = document.getElementById("runPredictionBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

const apiStatusText = document.querySelector(".api-status span:last-child");
const statusDot = document.querySelector(".status-dot");


function setApiStatus(isOnline) {
    if (isOnline) {
        statusDot.style.background = "#22c55e";
        apiStatusText.textContent = "API Online";
    } else {
        statusDot.style.background = "#ef4444";
        apiStatusText.textContent = "API Offline";
    }
}


async function checkApiHealth() {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/prediction/health`
        );

        if (!response.ok) {
            throw new Error("API health check failed");
        }

        const data = await response.json();

        setApiStatus(data.status === "success");

        return true;

    } catch (error) {
        console.error("API health error:", error);

        setApiStatus(false);

        return false;
    }
}


async function runPrediction() {

    runPredictionBtn.disabled = true;
    runPredictionBtn.textContent = "Running...";

    try {

        const response = await fetch(
            `${API_BASE_URL}/api/prediction`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

        const data = await response.json();

        if (!response.ok || data.status !== "success") {
            throw new Error(
                data.message || "Prediction request failed"
            );
        }

        displayPrediction(data);

        await loadPredictionHistory();

    } catch (error) {

        console.error("Prediction error:", error);

        alert(
            `Prediction failed: ${error.message}`
        );

    } finally {

        runPredictionBtn.disabled = false;
        runPredictionBtn.textContent = "Run Prediction";
    }
}


function displayPrediction(data) {

    const result = data.result || {};

    const confidence = result.confidence || {};
    const ensemble = result.ensemble || {};
    const agreement = result.model_agreement || {};
    const models = result.models || [];


    // Main prediction

    const predictedValue =
        result.prediction ?? "--";

    document.getElementById("currentPrediction").textContent =
        predictedValue;

    document.getElementById("predictionValue").textContent =
        predictedValue;


    // Prediction date

    document.getElementById("predictionDate").textContent =
        `Prediction date: ${
            data.prediction_date ??
            data.prediction?.prediction_date ??
            "--"
        }`;


    // Target

    document.getElementById("targetColumn").textContent =
        result.target_column ?? "--";


    // Ensemble probability

    document.getElementById("ensembleProbability").textContent =
        ensemble.probability != null
            ? `${Number(ensemble.probability).toFixed(2)}%`
            : "--";


    // Confidence

    document.getElementById("confidenceScore").textContent =
        confidence.score != null
            ? `${Number(confidence.score).toFixed(2)}%`
            : "--";


    document.getElementById("confidenceCategory").textContent =
        confidence.category ?? "--";


    // Model agreement

    document.getElementById("modelAgreement").textContent =
        agreement.agreement_count != null &&
        agreement.total_models != null
            ? `${agreement.agreement_count}/${agreement.total_models}`
            : "--";


    // Unique predictions

    document.getElementById("uniquePredictions").textContent =
        agreement.unique_predictions ?? "--";


    // Individual models

    displayModelPredictions(models);


    // Ensemble probabilities

    displayProbabilities(
        ensemble.probabilities || {}
    );
}


function displayModelPredictions(models) {

    const modelMap = {};

    models.forEach((model) => {

        if (
            model.model_name &&
            model.prediction !== undefined
        ) {

            modelMap[
                model.model_name
            ] = model.prediction;
        }

    });


    document.getElementById("randomForest").textContent =
        getModelPrediction(
            modelMap,
            "Random Forest"
        );


    document.getElementById("xgboost").textContent =
        getModelPrediction(
            modelMap,
            "XGBoost"
        );


    document.getElementById("lightgbm").textContent =
        getModelPrediction(
            modelMap,
            "LightGBM"
        );


    document.getElementById("catboost").textContent =
        getModelPrediction(
            modelMap,
            "CatBoost"
        );
}


function getModelPrediction(
    models,
    modelName
) {

    if (models[modelName] !== undefined) {
        return models[modelName];
    }


    const normalizedName =
        modelName
            .toLowerCase()
            .replace(/\s+/g, "");


    for (const key of Object.keys(models)) {

        const normalizedKey =
            key
                .toLowerCase()
                .replace(/\s+/g, "");


        if (normalizedKey === normalizedName) {
            return models[key];
        }
    }


    return "--";
}


function displayProbabilities(probabilities) {

    const probabilityItems =
        document.querySelectorAll(
            ".probability-item"
        );


    probabilityItems.forEach((item) => {

        const value =
            item
                .querySelector("span")
                .textContent;


        const probability =
            probabilities[value];


        item.querySelector("strong").textContent =
            probability != null
                ? `${Number(probability).toFixed(2)}%`
                : "--";
    });
}


async function loadPredictionHistory() {

    try {

        const response = await fetch(
            `${API_BASE_URL}/api/prediction/history`
        );


        if (!response.ok) {
            throw new Error(
                "Failed to load prediction history"
            );
        }


        const data =
            await response.json();


        if (data.status !== "success") {
            throw new Error(
                data.message ||
                "History request failed"
            );
        }


        document.getElementById(
            "totalRecords"
        ).textContent =
            data.total_records ?? 0;


        displayHistory(
            data.predictions || []
        );


    } catch (error) {

        console.error(
            "History error:",
            error
        );
    }
}


function displayHistory(predictions) {

    const tableBody =
        document.getElementById(
            "historyTableBody"
        );


    if (!predictions.length) {

        tableBody.innerHTML = `
            <tr>
                <td colspan="7">
                    No prediction history found.
                </td>
            </tr>
        `;

        return;
    }


    tableBody.innerHTML =
        predictions.map(
            (prediction) => {

                return `
                    <tr>

                        <td>
                            ${prediction.id ?? "--"}
                        </td>

                        <td>
                            ${
                                prediction.prediction_date ??
                                "--"
                            }
                        </td>

                        <td>
                            ${
                                prediction.target_column ??
                                "--"
                            }
                        </td>

                        <td>
                            ${
                                prediction.predicted_value ??
                                "--"
                            }
                        </td>

                        <td>
                            ${
                                prediction.ensemble_probability != null
                                    ? `${Number(
                                        prediction.ensemble_probability
                                    ).toFixed(2)}%`
                                    : "--"
                            }
                        </td>

                        <td>
                            ${
                                prediction.confidence_score != null
                                    ? `${Number(
                                        prediction.confidence_score
                                    ).toFixed(2)}%`
                                    : "--"
                            }
                        </td>

                        <td>
                            ${
                                prediction.prediction_status ??
                                "--"
                            }
                        </td>

                    </tr>
                `;
            }
        ).join("");
}


runPredictionBtn.addEventListener(
    "click",
    runPrediction
);


refreshHistoryBtn.addEventListener(
    "click",
    loadPredictionHistory
);


async function initializeDashboard() {

    const apiOnline =
        await checkApiHealth();


    if (apiOnline) {

        await loadPredictionHistory();

    }
}


initializeDashboard();