import qinj_analysis_helper as helper

helper.toSingleDataFramePerDirectory_newEventModel(
    path_to_dir = '../../tmp/money_plot_ET2p01_Bar4_m25c',
    dir_name_pattern = 'ET2p01_Bar4_VRef_SCurve_TDC_Pixel*',
    # save_to_csv = True,
    # debugging = True
)