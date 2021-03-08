from collections import OrderedDict

class server_global():

    def __init__(self):
        self.server_model_id = 0
        self.server_model = OrderedDict
        self.models_from_clients = []
        self.num_model_from_clients = 0
        self.recv_model_enabel = True
        self.max_models_allow = 10

    def get_server_model_id(self):
        return self.server_model_id

    def get_server_model(self):
        return self.server_model

    def get_num_model_from_clients(self):
        return self.num_model_from_clients

    def get_models_from_clients(self):
        return self.models_from_clients.copy()

    def get_recv_model_enabel(self):
        return self.recv_model_enabel

    def update_server_model_id(self):
        self.server_model_id += 1

    def set_max_models(self, max):
        self.max_models_allow = max

    def set_server_model(self, model):
        self.server_model = model.copy()

    def clear_models_from_clients(self):
        self.models_from_clients.clear()
        self.num_model_from_clients = 0

    def set_recv_model_enabel(self, stata):
        self.recv_model_enabel = stata

    def add_model_from_client(self, model):
        self.models_from_clients.append(model)
        self.num_model_from_clients += 1
        self.recv_model_enabel = self.num_model_from_clients < self.max_models_allow