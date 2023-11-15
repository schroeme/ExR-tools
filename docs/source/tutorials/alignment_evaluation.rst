3- Alignment Evaluation
========================

This tutorial describes the process of evaluating alignment in the ExR-Tools, including measuring alignment accuracy and calculating confidence intervals.

Step 1: Load Configuration Settings
------------------------------------

Start by loading the configuration settings.

.. code-block:: python

    from exr.config.config import Config

    # Create a new Config object instance.
    config = Config()

    # Provide the path to the configuration file.
    config_file_path = '/path/to/your/configuration/file.json'

    # Load the configuration settings from the specified file.
    config.load_config(config_file_path)

Step 2: Additional Configuration for Alignment Evaluation
---------------------------------------------------------

Configure additional parameters for alignment evaluation.

.. code-block:: python

    # Set various parameters for alignment evaluation
    config.nonzero_thresh = .2 * 2048 * 2048 * 80
    config.N = 1000
    config.subvol_dim = 100
    config.xystep = 0.1625/40  # check value
    config.zstep = 0.25/40  # check value
    config.pct_thresh = 99

Step 3: Alignment Measurement
-----------------------------

Measure alignment for specified rounds and ROIs.

.. code-block:: python

    from exr.align.align_eval import measure_round_alignment_NCC

    round_to_analyze = config.rounds
    roi_to_analyze = config.rois  # Adjust based on your dataset.

    for roi in roi_to_analyze:
        for round in round_to_analyze:
            measure_round_alignment_NCC(config=config, round=round, roi=roi)

Step 4: Alignment Evaluation and Confidence Interval Calculation
----------------------------------------------------------------

Evaluate alignment and calculate confidence intervals.

.. code-block:: python

    from exr.align.align_eval import plot_alignment_evaluation, calculate_alignment_evaluation_ci

    ci_percentage = 95
    percentile_filter_value = 95

    for roi in roi_to_analyze:
        plot_alignment_evaluation(config, roi, percentile=percentile_filter_value, save_fig=True)
        calculate_alignment_evaluation_ci(config, roi, ci=ci_percentage, percentile_filter=percentile_filter_value)

Next Steps
----------

After assessing the alignment, the next step in the ExSeq-Toolbox workflow is *Synapses Segmentation*. This step involves segmenting and analyzing synapses within the dataset. For detailed instructions on Synapses Segmentation, refer to the `Synapses Segmentation <synapses_segmentation.html>`_ section of this guide.
