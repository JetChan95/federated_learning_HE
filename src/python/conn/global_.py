class server_global():

    def __init__(self):
        self.server_model = None
        self.models_from_clients = []
        self.num_model_from_clients = 0
        self.recv_model_enabel = True

    def get_server_model(self):
        return self.server_model

    def get_num_model_from_clients(self):
        return self.num_model_from_clients

    def get_models_from_clients(self):
        return self.models_from_clients.copy()

    def get_recv_model_enabel(self):
        return self.recv_model_enabel

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