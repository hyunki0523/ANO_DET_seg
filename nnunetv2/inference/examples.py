nnUNet_raw=" /mnt/nas100/forGPU2/SungchulOn/_pred/nii"
nnUNet_results=" /mnt/nas100/forGPU2/SungchulOn/_pred/pred"

if __name__ == '__main__':
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        '/mnt/nas100/forGPU2/SungchulOn/nnUNet_v2/nnUNet_results/Dataset500_dental/nnUNetTrainer__nnUNetPlans__3d_fullres',
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files('/mnt/nas100/forGPU2/SungchulOn/_pred/nii',
                                '/mnt/nas100/forGPU2/SungchulOn/_pred/pred',
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    indir = '/mnt/nas100/forGPU2/SungchulOn/_pred/nii/'
    outdir = '/mnt/nas100/forGPU2/SungchulOn/_pred/pred/'
    predictor.predict_from_files('/mnt/nas100/forGPU2/SungchulOn/_pred/nii',
                                 '/mnt/nas100/forGPU2/SungchulOn/_pred/pred',
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2.5, returns segmentations
    indir = '/mnt/nas100/forGPU2/SungchulOn/test/nii/'
    predicted_segmentations = predictor.predict_from_files([[join(indir, 'test_01_0000.nii.gz')],
                                                            [join(indir, 'test_01_0000.nii.gz')]],
                                                           None,
                                                           save_probabilities=True, overwrite=True,
                                                           num_processes_preprocessing=2,
                                                           num_processes_segmentation_export=2,
                                                           folder_with_segs_from_prev_stage=None, num_parts=1,
                                                           part_id=0)

    # each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
    # If 'ofile' is None, the result will be returned instead of written to a file
    # the iterator is responsible for performing the correct preprocessing!
    # note how the iterator here does not use multiprocessing -> preprocessing will be done in the main thread!
    # take a look at the default iterators for predict_from_files and predict_from_list_of_npy_arrays
    # (they both use predictor.predict_from_data_iterator) for inspiration!
    def my_iterator(list_of_input_arrs, list_of_input_props):
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
        for a, p in zip(list_of_input_arrs, list_of_input_props):
            data, seg = preprocessor.run_case_npy(a,
                                                  None,
                                                  p,
                                                  predictor.plans_manager,
                                                  predictor.configuration_manager,
                                                  predictor.dataset_json)
            yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': p, 'ofile': None}

