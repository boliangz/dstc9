import copy
import os
from flask import Flask, request, jsonify
from queue import PriorityQueue
from threading import Thread

# Agent
from end2end.soloist.methods.baseline.SoloistAgent import Soloist

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)

# soloist model
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
agent = Soloist(model_file=model_dir, user_simulator=False)

print(agent.response('I am looking for a hotel'))


@app.route('/', methods=['GET', 'POST'])
def process():
  try:
    in_request = request.json
    print(in_request)
  except:
    return "invalid input: {}".format(in_request)
  rgi_queue.put(in_request)
  rgi_queue.join()
  output = rgo_queue.get()
  print(output['response'])
  rgo_queue.task_done()
  # return jsonify({'response': response})
  return jsonify(output)


def generate_response(in_queue, out_queue):
  while True:
    in_request = in_queue.get()
    user_utterance = in_request['input']
    agent_state = in_request['agent_state']

    if agent_state == {}:
      agent.init_session()
      agent_state = []

    agent_response, agent_state = agent.response(
      user_utterance, agent_state
    )

    out_queue.put({'response': agent_response, 'agent_state': agent_state})
    in_queue.task_done()
    out_queue.join()


if __name__ == '__main__':
  worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
  worker.setDaemon(True)
  worker.start()

  app.run(host='0.0.0.0', port=10004)
