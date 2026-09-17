class CyphaReasoning:
    def plan(self, goal, steps=3):
        return [f"Step {i+1}: achieve subgoal" for i in range(steps)] + [f"Finish: {goal}"]
    def analogical(self, A, B):
        if type(A) == type(B):
            return True
        return False
    def counterfactual(self, fact, hypothesis):
        return f"If {hypothesis}, then {fact}?"