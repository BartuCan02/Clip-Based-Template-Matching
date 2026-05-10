from .criterions_TM import SetCriterion_TM

def build_criterion(args):
    criterion = SetCriterion_TM(args.focal_loss)

    return criterion