??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-6-g2dd7e988??
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
??*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:?*
dtype0
p
	d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*
shared_name	d2/kernel
i
d2/kernel/Read/ReadVariableOpReadVariableOp	d2/kernel* 
_output_shapes
:
?	?*
dtype0
g
d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	d2/bias
`
d2/bias/Read/ReadVariableOpReadVariableOpd2/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer_with_weights-2
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
R
(regularization_losses
)	variables
*trainable_variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
 
 
*
0
1
,2
-3
64
75
*
0
1
,2
-3
64
75
?
<non_trainable_variables
=metrics
regularization_losses
>layer_regularization_losses

?layers
	variables
trainable_variables
 
 
 
 
?
@non_trainable_variables
Ametrics
regularization_losses
Blayer_regularization_losses

Clayers
	variables
trainable_variables
 
 
 
?
Dnon_trainable_variables
Emetrics
regularization_losses
Flayer_regularization_losses

Glayers
	variables
trainable_variables
 
 
 
?
Hnon_trainable_variables
Imetrics
regularization_losses
Jlayer_regularization_losses

Klayers
	variables
trainable_variables
US
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Lnon_trainable_variables
Mmetrics
 regularization_losses
Nlayer_regularization_losses

Olayers
!	variables
"trainable_variables
 
 
 
?
Pnon_trainable_variables
Qmetrics
$regularization_losses
Rlayer_regularization_losses

Slayers
%	variables
&trainable_variables
 
 
 
?
Tnon_trainable_variables
Umetrics
(regularization_losses
Vlayer_regularization_losses

Wlayers
)	variables
*trainable_variables
US
VARIABLE_VALUE	d2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
Xnon_trainable_variables
Ymetrics
.regularization_losses
Zlayer_regularization_losses

[layers
/	variables
0trainable_variables
 
 
 
?
\non_trainable_variables
]metrics
2regularization_losses
^layer_regularization_losses

_layers
3	variables
4trainable_variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
`non_trainable_variables
ametrics
8regularization_losses
blayer_regularization_losses

clayers
9	variables
:trainable_variables
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_mhc_inputPlaceholder*+
_output_shapes
:?????????"*
dtype0* 
shape:?????????"
?
serving_default_peptide_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_mhc_inputserving_default_peptide_input	d1/kerneld1/bias	d2/kerneld2/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*.
f)R'
%__inference_signature_wrapper_2983779
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd2/kernel/Read/ReadVariableOpd2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__traced_save_2984148
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d2/kerneld2/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference__traced_restore_2984178ζ
?%
?
B__inference_model_layer_call_and_return_conditional_losses_2983695
	mhc_input
peptide_input%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_29834692
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_29834832
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_concat_layer_call_and_return_conditional_losses_29834982
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d1_layer_call_and_return_conditional_losses_29835192
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835562
dropout_d1/PartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_skip1_layer_call_and_return_conditional_losses_29835762
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d2_layer_call_and_return_conditional_losses_29835972
d2/StatefulPartitionedCall?
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836342
dropout_d2/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_29836582 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?	
?
'__inference_model_layer_call_fn_2983909
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_29837552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
$__inference_d2_layer_call_fn_2984032

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d2_layer_call_and_return_conditional_losses_29835972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
o
C__inference_concat_layer_call_and_return_conditional_losses_2983938
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
e
,__inference_dropout_d2_layer_call_fn_2984062

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983551

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983556

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
G
+__inference_flatten_1_layer_call_fn_2983920

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_29834692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?
S
'__inference_skip1_layer_call_fn_2984012
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_skip1_layer_call_and_return_conditional_losses_29835762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
f
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984052

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2983629

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
C__inference_output_layer_call_and_return_conditional_losses_2984078

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
$__inference_d1_layer_call_fn_2983964

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d1_layer_call_and_return_conditional_losses_29835192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_output_layer_call_fn_2984085

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_29836582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
-
__inference_loss_fn_0_2984090
identity
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Constg
IdentityIdentity$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?

?
?__inference_d1_layer_call_and_return_conditional_losses_2983519

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_2983483

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?(
?
B__inference_model_layer_call_and_return_conditional_losses_2983673
	mhc_input
peptide_input%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?"dropout_d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_29834692
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_29834832
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_concat_layer_call_and_return_conditional_losses_29834982
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d1_layer_call_and_return_conditional_losses_29835192
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835512$
"dropout_d1/StatefulPartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_skip1_layer_call_and_return_conditional_losses_29835762
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d2_layer_call_and_return_conditional_losses_29835972
d2/StatefulPartitionedCall?
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836292$
"dropout_d2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_29836582 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?

?
?__inference_d2_layer_call_and_return_conditional_losses_2983597

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
m
C__inference_concat_layer_call_and_return_conditional_losses_2983498

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2983469

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2983915

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?(
?
B__inference_model_layer_call_and_return_conditional_losses_2983885
inputs_0
inputs_1%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
flatten/Const?
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis?
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat/concat?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d1/Relu?
dropout_d1/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_d1/Identityh
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis?
skip1/concatConcatV2concat/concat:output:0dropout_d1/Identity:output:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
skip1/concat?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d2/Relu?
dropout_d2/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_d2/Identity?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d2/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?+
?
"__inference__wrapped_model_2983458
	mhc_input
peptide_input+
'model_d1_matmul_readvariableop_resource,
(model_d1_biasadd_readvariableop_resource+
'model_d2_matmul_readvariableop_resource,
(model_d2_biasadd_readvariableop_resource/
+model_output_matmul_readvariableop_resource0
,model_output_biasadd_readvariableop_resource
identity??model/d1/BiasAdd/ReadVariableOp?model/d1/MatMul/ReadVariableOp?model/d2/BiasAdd/ReadVariableOp?model/d2/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape	mhc_inputmodel/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
model/flatten/Const?
model/flatten/ReshapeReshapepeptide_inputmodel/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshapev
model/concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat/concat/axis?
model/concat/concatConcatV2 model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0!model/concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concat/concat?
model/d1/MatMul/ReadVariableOpReadVariableOp'model_d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/d1/MatMul/ReadVariableOp?
model/d1/MatMulMatMulmodel/concat/concat:output:0&model/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d1/MatMul?
model/d1/BiasAdd/ReadVariableOpReadVariableOp(model_d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
model/d1/BiasAdd/ReadVariableOp?
model/d1/BiasAddBiasAddmodel/d1/MatMul:product:0'model/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d1/BiasAddt
model/d1/ReluRelumodel/d1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/d1/Relu?
model/dropout_d1/IdentityIdentitymodel/d1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_d1/Identityt
model/skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/skip1/concat/axis?
model/skip1/concatConcatV2model/concat/concat:output:0"model/dropout_d1/Identity:output:0 model/skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
model/skip1/concat?
model/d2/MatMul/ReadVariableOpReadVariableOp'model_d2_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
model/d2/MatMul/ReadVariableOp?
model/d2/MatMulMatMulmodel/skip1/concat:output:0&model/d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d2/MatMul?
model/d2/BiasAdd/ReadVariableOpReadVariableOp(model_d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
model/d2/BiasAdd/ReadVariableOp?
model/d2/BiasAddBiasAddmodel/d2/MatMul:product:0'model/d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d2/BiasAddt
model/d2/ReluRelumodel/d2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/d2/Relu?
model/dropout_d2/IdentityIdentitymodel/d2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_d2/Identity?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMul"model/dropout_d2/Identity:output:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/output/BiasAdd?
model/output/SigmoidSigmoidmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/output/Sigmoid?
IdentityIdentitymodel/output/Sigmoid:y:0 ^model/d1/BiasAdd/ReadVariableOp^model/d1/MatMul/ReadVariableOp ^model/d2/BiasAdd/ReadVariableOp^model/d2/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::2B
model/d1/BiasAdd/ReadVariableOpmodel/d1/BiasAdd/ReadVariableOp2@
model/d1/MatMul/ReadVariableOpmodel/d1/MatMul/ReadVariableOp2B
model/d2/BiasAdd/ReadVariableOpmodel/d2/BiasAdd/ReadVariableOp2@
model/d2/MatMul/ReadVariableOpmodel/d2/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?%
?
B__inference_model_layer_call_and_return_conditional_losses_2983755

inputs
inputs_1%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_29834692
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_29834832
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_concat_layer_call_and_return_conditional_losses_29834982
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d1_layer_call_and_return_conditional_losses_29835192
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835562
dropout_d1/PartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_skip1_layer_call_and_return_conditional_losses_29835762
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d2_layer_call_and_return_conditional_losses_29835972
d2/StatefulPartitionedCall?
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836342
dropout_d2/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_29836582 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?

?
?__inference_d1_layer_call_and_return_conditional_losses_2983957

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
'__inference_model_layer_call_fn_2983764
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_29837552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?	
?
'__inference_model_layer_call_fn_2983897
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_29837212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
H
,__inference_dropout_d2_layer_call_fn_2984067

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_2983926

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
C__inference_output_layer_call_and_return_conditional_losses_2983658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983989

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
!
cond_true_2984118
identityT
ConstConst*
_output_shapes
: *
dtype0*
valueB B.part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
l
B__inference_skip1_layer_call_and_return_conditional_losses_2983576

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?(
?
B__inference_model_layer_call_and_return_conditional_losses_2983721

inputs
inputs_1%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??d1/StatefulPartitionedCall?d2/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?"dropout_d2/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_29834692
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_29834832
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_concat_layer_call_and_return_conditional_losses_29834982
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d1_layer_call_and_return_conditional_losses_29835192
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835512$
"dropout_d1/StatefulPartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_skip1_layer_call_and_return_conditional_losses_29835762
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*H
fCRA
?__inference_d2_layer_call_and_return_conditional_losses_29835972
d2/StatefulPartitionedCall?
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d2_layer_call_and_return_conditional_losses_29836292$
"dropout_d2/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_29836582 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?

?
?__inference_d2_layer_call_and_return_conditional_losses_2984025

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
e
,__inference_dropout_d1_layer_call_fn_2983994

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
? 
?
 __inference__traced_save_2984148
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d2_kernel_read_readvariableop&
"savev2_d2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:0*
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatch?
condStatelessIfStaticRegexFullMatch:output:0"/device:CPU:0*
Tcond0
*	
Tin
 *
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *%
else_branchR
cond_false_2984119*
output_shapes
: *$
then_branchR
cond_true_29841182
condi
cond/IdentityIdentitycond:output:0"/device:CPU:0*
T0*
_output_shapes
: 2
cond/Identity{

StringJoin
StringJoinfile_prefixcond/Identity:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d2_kernel_read_readvariableop"savev2_d2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*N
_input_shapes=
;: :
??:?:
?	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?

?
'__inference_model_layer_call_fn_2983730
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_29837212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?P
?
B__inference_model_layer_call_and_return_conditional_losses_2983847
inputs_0
inputs_1%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
flatten/Const?
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis?
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat/concat?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d1/Reluw
dropout_d1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_d1/dropout/ratey
dropout_d1/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_d1/dropout/Shape?
%dropout_d1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d1/dropout/random_uniform/min?
%dropout_d1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_d1/dropout/random_uniform/max?
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_d1/dropout/random_uniform/RandomUniform?
%dropout_d1/dropout/random_uniform/subSub.dropout_d1/dropout/random_uniform/max:output:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d1/dropout/random_uniform/sub?
%dropout_d1/dropout/random_uniform/mulMul8dropout_d1/dropout/random_uniform/RandomUniform:output:0)dropout_d1/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2'
%dropout_d1/dropout/random_uniform/mul?
!dropout_d1/dropout/random_uniformAdd)dropout_d1/dropout/random_uniform/mul:z:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2#
!dropout_d1/dropout/random_uniformy
dropout_d1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d1/dropout/sub/x?
dropout_d1/dropout/subSub!dropout_d1/dropout/sub/x:output:0 dropout_d1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/sub?
dropout_d1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d1/dropout/truediv/x?
dropout_d1/dropout/truedivRealDiv%dropout_d1/dropout/truediv/x:output:0dropout_d1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/truediv?
dropout_d1/dropout/GreaterEqualGreaterEqual%dropout_d1/dropout/random_uniform:z:0 dropout_d1/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2!
dropout_d1/dropout/GreaterEqual?
dropout_d1/dropout/mulMuld1/Relu:activations:0dropout_d1/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_d1/dropout/mul?
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_d1/dropout/Cast?
dropout_d1/dropout/mul_1Muldropout_d1/dropout/mul:z:0dropout_d1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_d1/dropout/mul_1h
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis?
skip1/concatConcatV2concat/concat:output:0dropout_d1/dropout/mul_1:z:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
skip1/concat?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d2/Reluw
dropout_d2/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_d2/dropout/ratey
dropout_d2/dropout/ShapeShaped2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_d2/dropout/Shape?
%dropout_d2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d2/dropout/random_uniform/min?
%dropout_d2/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_d2/dropout/random_uniform/max?
/dropout_d2/dropout/random_uniform/RandomUniformRandomUniform!dropout_d2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_d2/dropout/random_uniform/RandomUniform?
%dropout_d2/dropout/random_uniform/subSub.dropout_d2/dropout/random_uniform/max:output:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d2/dropout/random_uniform/sub?
%dropout_d2/dropout/random_uniform/mulMul8dropout_d2/dropout/random_uniform/RandomUniform:output:0)dropout_d2/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2'
%dropout_d2/dropout/random_uniform/mul?
!dropout_d2/dropout/random_uniformAdd)dropout_d2/dropout/random_uniform/mul:z:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2#
!dropout_d2/dropout/random_uniformy
dropout_d2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d2/dropout/sub/x?
dropout_d2/dropout/subSub!dropout_d2/dropout/sub/x:output:0 dropout_d2/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/sub?
dropout_d2/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d2/dropout/truediv/x?
dropout_d2/dropout/truedivRealDiv%dropout_d2/dropout/truediv/x:output:0dropout_d2/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/truediv?
dropout_d2/dropout/GreaterEqualGreaterEqual%dropout_d2/dropout/random_uniform:z:0 dropout_d2/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2!
dropout_d2/dropout/GreaterEqual?
dropout_d2/dropout/mulMuld2/Relu:activations:0dropout_d2/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_d2/dropout/mul?
dropout_d2/dropout/CastCast#dropout_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_d2/dropout/Cast?
dropout_d2/dropout/mul_1Muldropout_d2/dropout/mul:z:0dropout_d2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_d2/dropout/mul_1?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d2/dropout/mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const?
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
e
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984057

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
-
__inference_loss_fn_1_2984095
identity
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Constg
IdentityIdentity$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
T
(__inference_concat_layer_call_fn_2983944
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_concat_layer_call_and_return_conditional_losses_29834982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
E
)__inference_flatten_layer_call_fn_2983931

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_29834832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout_d1_layer_call_fn_2983999

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_d1_layer_call_and_return_conditional_losses_29835562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
f
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983984

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2983634

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_2983779
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__wrapped_model_29834582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????":?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
n
B__inference_skip1_layer_call_and_return_conditional_losses_2984006
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
"
cond_false_2984119
identityz
ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8b57796f45a745eebd129851d844b879/part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
? 
?
#__inference__traced_restore_2984178
file_prefix
assignvariableop_d1_kernel
assignvariableop_1_d1_bias 
assignvariableop_2_d2_kernel
assignvariableop_3_d2_bias$
 assignvariableop_4_output_kernel"
assignvariableop_5_output_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_d2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_d2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
	mhc_input6
serving_default_mhc_input:0?????????"
K
peptide_input:
serving_default_peptide_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer_with_weights-2
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
d_default_save_signature
*e&call_and_return_all_conditional_losses
f__call__"?;
_tf_keras_model?;{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d2", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d2", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "loss", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0019147476238602517, "decay": 0.0, "rho": 0.9, "momentum": 0.5, "epsilon": 3.17051703095139e-07, "centered": true}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "mhc_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 34, 20], "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "peptide_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 15, 20], "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*k&call_and_return_all_conditional_losses
l__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}}
?

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*m&call_and_return_all_conditional_losses
n__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 980}}}}
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
(regularization_losses
)	variables
*trainable_variables
+	keras_api
*q&call_and_return_all_conditional_losses
r__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "skip1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}}
?

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "d2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1236}}}}
?
2regularization_losses
3	variables
4trainable_variables
5	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
"
	optimizer
.
y0
z1"
trackable_list_wrapper
J
0
1
,2
-3
64
75"
trackable_list_wrapper
J
0
1
,2
-3
64
75"
trackable_list_wrapper
?
<non_trainable_variables
=metrics
regularization_losses
>layer_regularization_losses

?layers
	variables
trainable_variables
f__call__
d_default_save_signature
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables
Ametrics
regularization_losses
Blayer_regularization_losses

Clayers
	variables
trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables
Emetrics
regularization_losses
Flayer_regularization_losses

Glayers
	variables
trainable_variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables
Imetrics
regularization_losses
Jlayer_regularization_losses

Klayers
	variables
trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:
??2	d1/kernel
:?2d1/bias
'
y0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Lnon_trainable_variables
Mmetrics
 regularization_losses
Nlayer_regularization_losses

Olayers
!	variables
"trainable_variables
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables
Qmetrics
$regularization_losses
Rlayer_regularization_losses

Slayers
%	variables
&trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables
Umetrics
(regularization_losses
Vlayer_regularization_losses

Wlayers
)	variables
*trainable_variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:
?	?2	d2/kernel
:?2d2/bias
'
z0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
Xnon_trainable_variables
Ymetrics
.regularization_losses
Zlayer_regularization_losses

[layers
/	variables
0trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables
]metrics
2regularization_losses
^layer_regularization_losses

_layers
3	variables
4trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 :	?2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
`non_trainable_variables
ametrics
8regularization_losses
blayer_regularization_losses

clayers
9	variables
:trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
"__inference__wrapped_model_2983458?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *^?[
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
?2?
B__inference_model_layer_call_and_return_conditional_losses_2983847
B__inference_model_layer_call_and_return_conditional_losses_2983885
B__inference_model_layer_call_and_return_conditional_losses_2983673
B__inference_model_layer_call_and_return_conditional_losses_2983695?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_layer_call_fn_2983909
'__inference_model_layer_call_fn_2983764
'__inference_model_layer_call_fn_2983897
'__inference_model_layer_call_fn_2983730?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_flatten_1_layer_call_and_return_conditional_losses_2983915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_1_layer_call_fn_2983920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_2983926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_layer_call_fn_2983931?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_concat_layer_call_and_return_conditional_losses_2983938?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_concat_layer_call_fn_2983944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_d1_layer_call_and_return_conditional_losses_2983957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_d1_layer_call_fn_2983964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983989
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983984?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_d1_layer_call_fn_2983999
,__inference_dropout_d1_layer_call_fn_2983994?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_skip1_layer_call_and_return_conditional_losses_2984006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_skip1_layer_call_fn_2984012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_d2_layer_call_and_return_conditional_losses_2984025?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_d2_layer_call_fn_2984032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984052
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984057?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dropout_d2_layer_call_fn_2984067
,__inference_dropout_d2_layer_call_fn_2984062?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_output_layer_call_and_return_conditional_losses_2984078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_output_layer_call_fn_2984085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_2984090?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_2984095?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
CBA
%__inference_signature_wrapper_2983779	mhc_inputpeptide_input?
"__inference__wrapped_model_2983458?,-67h?e
^?[
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
? "/?,
*
output ?
output??????????
C__inference_concat_layer_call_and_return_conditional_losses_2983938?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
(__inference_concat_layer_call_fn_2983944y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
?__inference_d1_layer_call_and_return_conditional_losses_2983957^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
$__inference_d1_layer_call_fn_2983964Q0?-
&?#
!?
inputs??????????
? "????????????
?__inference_d2_layer_call_and_return_conditional_losses_2984025^,-0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? y
$__inference_d2_layer_call_fn_2984032Q,-0?-
&?#
!?
inputs??????????	
? "????????????
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983984^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_d1_layer_call_and_return_conditional_losses_2983989^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_d1_layer_call_fn_2983994Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_d1_layer_call_fn_2983999Q4?1
*?'
!?
inputs??????????
p 
? "????????????
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984052^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
G__inference_dropout_d2_layer_call_and_return_conditional_losses_2984057^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
,__inference_dropout_d2_layer_call_fn_2984062Q4?1
*?'
!?
inputs??????????
p
? "????????????
,__inference_dropout_d2_layer_call_fn_2984067Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_flatten_1_layer_call_and_return_conditional_losses_2983915]3?0
)?&
$?!
inputs?????????"
? "&?#
?
0??????????
? 
+__inference_flatten_1_layer_call_fn_2983920P3?0
)?&
$?!
inputs?????????"
? "????????????
D__inference_flatten_layer_call_and_return_conditional_losses_2983926]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? }
)__inference_flatten_layer_call_fn_2983931P3?0
)?&
$?!
inputs?????????
? "???????????9
__inference_loss_fn_0_2984090?

? 
? "? 9
__inference_loss_fn_1_2984095?

? 
? "? ?
B__inference_model_layer_call_and_return_conditional_losses_2983673?,-67p?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2983695?,-67p?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2983847?,-67j?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2983885?,-67j?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
'__inference_model_layer_call_fn_2983730?,-67p?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p

 
? "???????????
'__inference_model_layer_call_fn_2983764?,-67p?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p 

 
? "???????????
'__inference_model_layer_call_fn_2983897?,-67j?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p

 
? "???????????
'__inference_model_layer_call_fn_2983909?,-67j?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p 

 
? "???????????
C__inference_output_layer_call_and_return_conditional_losses_2984078]670?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_output_layer_call_fn_2984085P670?-
&?#
!?
inputs??????????
? "???????????
%__inference_signature_wrapper_2983779?,-67??~
? 
w?t
4
	mhc_input'?$
	mhc_input?????????"
<
peptide_input+?(
peptide_input?????????"/?,
*
output ?
output??????????
B__inference_skip1_layer_call_and_return_conditional_losses_2984006?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????	
? ?
'__inference_skip1_layer_call_fn_2984012y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "???????????	