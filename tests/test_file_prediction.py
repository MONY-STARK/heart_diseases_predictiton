import io

def test_file_prediction(client):
    csv_content = """male,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose
1,50,1,10,0,0,1,0,240,140,90,27.5,75,85
"""

    file = {
        "file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")
    }

    response = client.post("/predict-file", files=file)

    assert response.status_code == 200
    result = response.json()

    assert isinstance(result, list)
    assert "Prediction" in result[0]
    assert "Risk_Probability" in result[0]
