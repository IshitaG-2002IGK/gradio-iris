
import gradio as gr
from pycaret.datasets import get_data

data = get_data('iris')

# initialize setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

best = compare_models(n_select = 15)
compare_model_results = pull()

# creating a predict function to be passed into gradio UI
def predict(model, sepal_length, sepal_width, petal_length, petal_width):
  
    df = pd.DataFrame.from_dict({'sepal_length': [sepal_length], 'sepal_width': [sepal_width], 
                                 'petal_length': [petal_length], 'petal_width': [petal_width]})
    
    model_index = list(compare_model_results['Model']).index(model)
    model = best[model_index]
    pred = predict_model(model, df, raw_score=True)
    return {'Iris-setosa': pred['Score_Iris-setosa'][0].astype('float64'), 
            'Iris-versicolor': pred['Score_Iris-versicolor'][0].astype('float64'),
            'Iris-virginica' : pred['Score_Iris-virginica'][0].astype('float64')}


import gradio as gr
model = gr.inputs.Dropdown(list(compare_model_results['Model']), label="Model")
sepal_length = gr.inputs.Slider(minimum=1, maximum=10, default=data['sepal_length'].mean(), label = 'sepal_length')
sepal_width = gr.inputs.Slider(minimum=1, maximum=10, default=data['sepal_width'].mean(), label = 'sepal_width')
petal_length = gr.inputs.Slider(minimum=1, maximum=10, default=data['petal_length'].mean(), label = 'petal_length')
petal_width = gr.inputs.Slider(minimum=1, maximum=10, default=data['petal_width'].mean(), label = 'petal_width')

gr.Interface(predict, [model,sepal_length,sepal_width,petal_length,petal_width], "label", live=True).launch()
