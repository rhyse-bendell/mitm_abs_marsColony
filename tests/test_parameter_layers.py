from modules.parameter_layers import ParameterLayer, classify_parameter


def test_parameter_layer_classification():
    assert classify_parameter("communication_propensity") == ParameterLayer.PILOT_CAPABILITY
    assert classify_parameter("artifact_consultation_tendency") == ParameterLayer.PILOT_CAPABILITY
    assert classify_parameter("dik_uptake_fidelity") == ParameterLayer.MECHA_CAPABILITY
    assert classify_parameter("site_b_capacity") == ParameterLayer.ENVIRONMENT_TASK
    assert classify_parameter("bridge_bc_cost") == ParameterLayer.ENVIRONMENT_TASK
