from roboflow import Roboflow
rf = Roboflow(api_key="wv1RWb130ECTfhxSNxPS")
project = rf.workspace("footdiseaseimgclass").project("things-jam67")
dataset = project.version(8).download("folder")
