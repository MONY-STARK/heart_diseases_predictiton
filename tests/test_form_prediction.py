def test_form_prediction(client, valid_input):
    response = client.post(
        "/predict-form",
        data=valid_input
    )

    assert response.status_code == 200
    json_data = response.json()

    assert "prediction" in json_data
    assert "probability" in json_data
    assert json_data["prediction"] in ["High Risk", "Low Risk"]
