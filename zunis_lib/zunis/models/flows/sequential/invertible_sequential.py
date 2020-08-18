import torch

from zunis.models.flows.general_flow import GeneralFlow


class InvertibleSequentialFlow(GeneralFlow):
    def __init__(self,d,flows):
        super(InvertibleSequentialFlow, self).__init__(d=d)

        self.flows = torch.nn.ModuleList(flows)

        for flow in flows:
            try:
                if not flow.runs_forward():
                    flow.invert()
            except AttributeError as e:
                print("All flows in an InvertibleSequentialFlow must be invertible")
                raise
        self.inverse = False

    def invert(self):
        self.inverse = not self.inverse
        for flow in self.flows:
            flow.invert()

    def flow(self, x):
        output = x
        if self.inverse:
            for f in self.flows[::-1]:
                output = f.flow(output)
        else:
            for f in self.flows:
                output = f.flow(output)

        return output


    def transform_and_compute_jacobian(self, xj):
        output = xj
        if self.inverse:
            for f in self.flows[::-1]:
                output = f.transform_and_compute_jacobian(output)
        else:
            for f in self.flows:
                output = f.transform_and_compute_jacobian(output)

        return output