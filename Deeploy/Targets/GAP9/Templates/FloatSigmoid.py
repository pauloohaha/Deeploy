# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Sigmoid from GAP9 SDK (Name: ${nodeName}, Op: ${nodeOp})
                                 
KerActivation_fp32_T sigmoid_arg;
sigmoid_arg.In    = (float *)${data_in};
sigmoid_arg.Out   = (float *)${data_out};
sigmoid_arg.Feat  = 1;
sigmoid_arg.W     = ${size};
sigmoid_arg.H     = 1;
                                 
KerParSigmoid_fp32(&sigmoid_arg);
""")
