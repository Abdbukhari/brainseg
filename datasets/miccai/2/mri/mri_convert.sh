#!/bin/bash

 for i in $( ls| grep .nii| cut -d '.' -f 1 ); do
            echo "Converting" $i".nii..."
            mri_convert $i".nii" $i".mgz"
        done