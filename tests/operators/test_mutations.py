import subprocess



def test_mutation():
    output = subprocess.check_output(['uv', 'run', 'src/llm_mutation.py', 'sota/ExquisiteNetV2/network.py', 'tests/operators/outputs/network_test.py', 'tests/operators/test_prompt_1.txt', '--top_p', '0.1', '--temperature', '0.17', '--apply_quality_control', 'False', '--inference_submission', 'True']) 
    print(output)   