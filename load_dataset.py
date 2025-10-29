from dataset_utils import built_data_and_save_for_splited_with_species_gender_exposure
args={}
args['input_csv'] = 'D:/data/aquatic data.csv'
args['output_bin'] = 'D:/data/aquatic data.bin'
args['output_csv'] = 'D:/data/aquatic data_group.csv'

built_data_and_save_for_splited_with_species_gender_exposure(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )