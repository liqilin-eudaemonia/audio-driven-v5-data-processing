import copy
import unittest
import torch
from p3_listen.training import restore_model


class FakeModel(torch.nn.Module):
    def __init__(self,stage):
        super().__init__()
        self.stage=stage
        self.planner_names=("heads.weight",)
        self.planner_parameters=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2,3))])
        if stage in ("C0","A"):
            self.connector=torch.nn.Linear(3,3)
        if stage=="A":
            self.adapters=torch.nn.Linear(3,3)


class CheckpointTests(unittest.TestCase):
    def test_transition_loads_planner_preserves_new_connector(self):
        p=FakeModel("P")
        with torch.no_grad():p.planner_parameters[0].fill_(2.)
        saved=dict(stage="P",planner_names=p.planner_names,model=p.state_dict())
        c=FakeModel("C0")
        initial=c.connector.weight.detach().clone()
        restore_model(c,saved,transition=True)
        self.assertTrue(torch.equal(c.planner_parameters[0],p.planner_parameters[0]))
        self.assertTrue(torch.equal(c.connector.weight,initial))
        with self.assertRaises(ValueError):restore_model(FakeModel("A"),saved,transition=True)

    def test_bad_checkpoint_validated_before_mutation(self):
        m=FakeModel("C0")
        before={k:v.clone() for k,v in m.state_dict().items()}
        bad=copy.deepcopy(before)
        bad["connector.bias"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            restore_model(m,dict(stage="C0",planner_names=m.planner_names,model=bad))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in m.state_dict().items()))


if __name__=="__main__":unittest.main()
