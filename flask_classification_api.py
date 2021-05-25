import os
import re
import json
import time
import traceback
import numpy as np

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, render_template

from training.text_classification import ClassifyText
text_classification_obj = ClassifyText()

# for reproducible results
np.random.seed(777)

app = Flask(__name__)
app.debug = True
app.url_map.strict_slashes = False
Swagger(app)


@app.route("/classification", methods=['POST', 'OPTIONS'])
@swag_from("swagger_files/classification_api.yml")
def classification():
	res_json = {"result": {}, "status":"", "model_id":"", "lang":""}
	try:
		st = time.time()
		body_data = json.loads(request.get_data())
		print("\n keys received --- ",body_data.keys())

		""" read input request """ 
		training_data_path = body_data["TrainingDataPath"]
		body_data = json.load(open(training_data_path, "r"))
		text_data = body_data["TrainingData"]
		lang = body_data["Language"]
		model_id = body_data["ModelId"]
		embedding_model = body_data["EmbeddingModel"]

		print("\n language --- ",lang)
		print("\n model_id --- ", model_id)
		print("\n embedding_model --- ",embedding_model)

		""" train clustering model """
		classification_dict = text_classification_obj.main(text_data, lang, embedding_model, model_id)
		res_json["result"] = json.dumps(clusters_dict, indent=4)
		res_json["model_id"] = model_id
		res_json["lang"] = lang
		res_json["status"] = "200"
		print("\n total time --- ",time.time() - st)

	except Exception as e:
		logger.error("\n Error in classification app main() :",traceback.format_exc())
		res_json["status"] = "400"

	# return render_template("index.html")
	return jsonify(res_json)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5001)
