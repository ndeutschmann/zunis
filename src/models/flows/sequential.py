from .general_flow import GeneralFlow


class InvertibleSequentialFlow(GeneralFlow):
    def __init__(self,d,flows):
        super(InvertibleSequentialFlow, self).__init__(d=d)

        self.flows = list(flows)

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
        self.flows.reverse()
        for flow in self.flows:
            flow.invert()

    def flow(self, x):
        output = x
        for f in self.flows:
            output = f.flow(output)
        return output

    def transform_and_compute_jacobian(self, xj):
        output = xj
        for f in self.flows:
            output = f.transform_and_compute_jacobian(output)
        return output
