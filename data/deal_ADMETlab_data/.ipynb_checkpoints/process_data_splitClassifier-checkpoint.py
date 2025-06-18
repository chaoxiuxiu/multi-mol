from data.deal_ADMETlab_data.utils import smiles2image_save_as_json_by_one_file
#from utils import smiles2image_save_as_json_by_one_file
if __name__ == '__main__':
    smiles2image_save_as_json_by_one_file(path='./data/ADMETlab_data/CycPeptMPDB_Peptide_Assay_PAMPA(4).csv',
                                          image_out_dir='./data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide/images',
                                          json_out_dir='./data/downstreamDataset/ADMETlab_data/CycPeptMPDB_Peptide')