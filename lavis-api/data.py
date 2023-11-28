from lavis.models import load_model_and_preprocess


global device
global model
global vis_processors
'''
device = "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
   model_type="vicuna7b",
    is_eval=True,
    device=device)
print(f"Loaded model to: {device}")
'''