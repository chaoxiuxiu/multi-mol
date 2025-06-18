from data.write_arrow.utils import make_arrow

# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            data_split_path='train.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            split = 'train'
#            )
# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            data_split_path='test.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            split = 'test'
#            )
# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            data_split_path='dev.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_0',
#            split = 'dev'
#            )
#
# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            data_split_path='train.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            split = 'train'
#            )
# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            data_split_path='test.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            split = 'test'
#            )
# make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            data_split_path='dev.json',
#            dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/clintox_1',
#            split = 'dev'
#            )

make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           data_split_path='train.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           split = 'train'
           )
make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           data_split_path='test.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           split = 'test'
           )
make_arrow(root='/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           data_split_path='dev.json',
           dataset_root = '/data1/zx/myCLIP/data/downstreamDataset/MoleculeNet/ClinTox',
           split = 'dev'
           )

print('done!')