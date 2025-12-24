def test_model_loaded(model):
    assert model is not None


def test_model_prediction_shape(model):
    import pandas as pd

    X = pd.DataFrame([{
        "male": 1,
        "age": 50,
        "currentSmoker": 1,
        "cigsPerDay": 10,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 240,
        "sysBP": 140,
        "diaBP": 90,
        "BMI": 27.5,
        "heartRate": 75,
        "glucose": 85
    }])

    pred = model.predict(X)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
