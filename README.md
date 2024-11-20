# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


* Task 2.5


* Train with CPU

* Simple Dataset

Ouput training logs:

Epoch  0  loss  5.475623812728197 correct 36  time per epoch 25.345622301101685
Epoch  10  loss  0.6180821346159754 correct 49  time per epoch 0.1347179412841797
Epoch  20  loss  0.6816592859530328 correct 50  time per epoch 0.3078012466430664
Epoch  30  loss  0.3979392128414974 correct 50  time per epoch 0.13416647911071777
Epoch  40  loss  0.13925991871216314 correct 50  time per epoch 0.13191723823547363
Epoch  50  loss  0.7773315026758553 correct 50  time per epoch 0.13302350044250488
Epoch  60  loss  0.07497553487122467 correct 50  time per epoch 0.13289117813110352
Epoch  70  loss  0.7834742071212054 correct 50  time per epoch 0.13256382942199707
Epoch  80  loss  0.0955282664094295 correct 50  time per epoch 0.13344383239746094
Epoch  90  loss  0.0550294736709935 correct 50  time per epoch 0.14815783500671387
Epoch  100  loss  0.5906038814808073 correct 50  time per epoch 0.1347954273223877
Epoch  110  loss  0.025901978806964154 correct 50  time per epoch 0.2363584041595459
Epoch  120  loss  0.15063332784487166 correct 50  time per epoch 0.1386418342590332
Epoch  130  loss  0.27017022688079667 correct 50  time per epoch 0.1321120262145996
Epoch  140  loss  0.025343026735077136 correct 50  time per epoch 0.13663983345031738
Epoch  150  loss  0.01825069342989664 correct 50  time per epoch 0.1316530704498291
Epoch  160  loss  0.028455716906017073 correct 50  time per epoch 0.1298818588256836
Epoch  170  loss  0.0077437495627934606 correct 50  time per epoch 0.13561797142028809
Epoch  180  loss  0.0777442793387418 correct 50  time per epoch 0.1351919174194336
Epoch  190  loss  0.5240371631874564 correct 50  time per epoch 0.19258904457092285
Epoch  200  loss  0.017134269817171 correct 50  time per epoch 0.18059635162353516
Epoch  210  loss  0.10966872447295174 correct 50  time per epoch 0.1328110694885254
Epoch  220  loss  0.13197095089754288 correct 50  time per epoch 0.1349043846130371
Epoch  230  loss  0.3470902865209392 correct 50  time per epoch 0.1334853172302246
Epoch  240  loss  0.5020570184505778 correct 50  time per epoch 0.13327908515930176
Epoch  250  loss  0.09649735763097271 correct 50  time per epoch 0.1405634880065918
Epoch  260  loss  0.009918928554434775 correct 50  time per epoch 0.13329648971557617
Epoch  270  loss  0.005526823125857086 correct 50  time per epoch 0.13450360298156738
Epoch  280  loss  0.055554803461745406 correct 50  time per epoch 0.3126962184906006
Epoch  290  loss  0.2243070783861036 correct 50  time per epoch 0.14266371726989746
Epoch  300  loss  0.35149141454566396 correct 50  time per epoch 0.13565421104431152
Epoch  310  loss  0.0030851627417632327 correct 50  time per epoch 0.13492321968078613
Epoch  320  loss  8.718921269614934e-05 correct 50  time per epoch 0.1367177963256836
Epoch  330  loss  0.0032920321744490683 correct 50  time per epoch 0.13399386405944824
Epoch  340  loss  0.39497236885838166 correct 50  time per epoch 0.1375730037689209
Epoch  350  loss  0.2621802000250292 correct 50  time per epoch 0.1408078670501709
Epoch  360  loss  0.0918491694601602 correct 50  time per epoch 0.13311290740966797
Epoch  370  loss  0.00276767880110797 correct 50  time per epoch 0.2429511547088623
Epoch  380  loss  0.4016349716776128 correct 50  time per epoch 0.13478970527648926
Epoch  390  loss  0.09803479448218516 correct 50  time per epoch 0.13301730155944824
Epoch  400  loss  0.2721228952181447 correct 50  time per epoch 0.14439797401428223
Epoch  410  loss  0.22094472841912094 correct 50  time per epoch 0.1322476863861084
Epoch  420  loss  0.05129012785221094 correct 50  time per epoch 0.13341856002807617
Epoch  430  loss  0.030109397514757708 correct 50  time per epoch 0.13289332389831543
Epoch  440  loss  0.0028609018161706045 correct 50  time per epoch 0.14946627616882324
Epoch  450  loss  0.0002312074970658428 correct 50  time per epoch 0.18968510627746582
Epoch  460  loss  0.2522607876858137 correct 50  time per epoch 0.1918327808380127
Epoch  470  loss  0.2035921831619263 correct 50  time per epoch 0.1334853172302246
Epoch  480  loss  0.29472402041524287 correct 50  time per epoch 0.1455678939819336
Epoch  490  loss  0.08355205045675271 correct 50  time per epoch 0.1327075958251953


* Split Dataset

Ouput training logs:

Epoch  0  loss  6.063230897042813 correct 33  time per epoch 25.44927668571472
Epoch  10  loss  5.119946262796765 correct 34  time per epoch 0.1405503749847412
Epoch  20  loss  6.774578191757526 correct 44  time per epoch 0.15424489974975586
Epoch  30  loss  5.889316406615169 correct 43  time per epoch 0.1428060531616211
Epoch  40  loss  4.495061526351483 correct 40  time per epoch 0.13354802131652832
Epoch  50  loss  4.730964152450551 correct 41  time per epoch 0.27086710929870605
Epoch  60  loss  4.359700400286146 correct 44  time per epoch 0.13325715065002441
Epoch  70  loss  3.255185285032586 correct 47  time per epoch 0.13711094856262207
Epoch  80  loss  1.8013327235312673 correct 46  time per epoch 0.14672565460205078
Epoch  90  loss  3.2559263888234007 correct 45  time per epoch 0.13850760459899902
Epoch  100  loss  2.2406887900706716 correct 46  time per epoch 0.14026284217834473
Epoch  110  loss  1.6941800940981275 correct 48  time per epoch 0.14424490928649902
Epoch  120  loss  2.6482026716255924 correct 47  time per epoch 0.13759899139404297
Epoch  130  loss  2.1513027919116867 correct 48  time per epoch 0.27247190475463867
Epoch  140  loss  1.5882342130727782 correct 45  time per epoch 0.136627197265625
Epoch  150  loss  4.762918649977745 correct 44  time per epoch 0.13435769081115723
Epoch  160  loss  1.4333480159607728 correct 48  time per epoch 0.133131742477417
Epoch  170  loss  0.7006862619286487 correct 46  time per epoch 0.13654112815856934
Epoch  180  loss  3.270227253249018 correct 48  time per epoch 0.13572907447814941
Epoch  190  loss  1.1812390225759886 correct 48  time per epoch 0.13415098190307617
Epoch  200  loss  2.097810723794338 correct 49  time per epoch 0.14881634712219238
Epoch  210  loss  1.2929367710955626 correct 48  time per epoch 0.13284969329833984
Epoch  220  loss  0.2006800326940556 correct 47  time per epoch 0.29773974418640137
Epoch  230  loss  1.2399701750349825 correct 47  time per epoch 0.1350095272064209
Epoch  240  loss  1.7671632623043019 correct 49  time per epoch 0.1324779987335205
Epoch  250  loss  0.728820323045602 correct 48  time per epoch 0.22075581550598145
Epoch  260  loss  1.0327133333834035 correct 48  time per epoch 0.13645005226135254
Epoch  270  loss  0.8887705902100896 correct 48  time per epoch 0.14850187301635742
Epoch  280  loss  2.382367386785893 correct 45  time per epoch 0.13324594497680664
Epoch  290  loss  1.7781229648515497 correct 47  time per epoch 0.13626384735107422
Epoch  300  loss  0.48721775696259484 correct 49  time per epoch 0.2032465934753418
Epoch  310  loss  1.2557081103027081 correct 48  time per epoch 0.1336216926574707
Epoch  320  loss  0.7750000653045823 correct 48  time per epoch 0.13896727561950684
Epoch  330  loss  1.3898137095349956 correct 48  time per epoch 0.13457012176513672
Epoch  340  loss  0.8548771466926569 correct 49  time per epoch 0.1330094337463379
Epoch  350  loss  2.2938908131345657 correct 48  time per epoch 0.1358487606048584
Epoch  360  loss  1.8228998150334934 correct 49  time per epoch 0.13776922225952148
Epoch  370  loss  3.5446020122960555 correct 47  time per epoch 0.13242554664611816
Epoch  380  loss  1.7500742619590492 correct 48  time per epoch 0.20834946632385254
Epoch  390  loss  0.5896555749871799 correct 49  time per epoch 0.13898420333862305
Epoch  400  loss  1.3608100596777248 correct 48  time per epoch 0.13305354118347168
Epoch  410  loss  1.3958531075814813 correct 48  time per epoch 0.13352108001708984
Epoch  420  loss  1.1864708667818644 correct 48  time per epoch 0.14999651908874512
Epoch  430  loss  1.8802797423273454 correct 50  time per epoch 0.13384771347045898
Epoch  440  loss  0.06046040848036936 correct 48  time per epoch 0.13489341735839844
Epoch  450  loss  0.0720853142554259 correct 49  time per epoch 0.1355748176574707
Epoch  460  loss  1.468283124934971 correct 49  time per epoch 0.15011334419250488
Epoch  470  loss  1.4030517303728527 correct 49  time per epoch 0.30833983421325684
Epoch  480  loss  4.016851892094923 correct 49  time per epoch 0.13621139526367188
Epoch  490  loss  1.6052413654416728 correct 49  time per epoch 0.13196253776550293

* Xor Dataset

Ouput training logs:

Epoch  0  loss  7.634667629543065 correct 31  time per epoch 24.960493803024292
Epoch  10  loss  5.209705090421572 correct 39  time per epoch 0.16501641273498535
Epoch  20  loss  5.495530599081412 correct 41  time per epoch 0.13608813285827637
Epoch  30  loss  4.9560499871417765 correct 40  time per epoch 0.13376307487487793
Epoch  40  loss  3.1663759231647215 correct 43  time per epoch 0.13414549827575684
Epoch  50  loss  1.6621688390879823 correct 43  time per epoch 0.13616061210632324
Epoch  60  loss  1.793638509207756 correct 41  time per epoch 0.15828251838684082
Epoch  70  loss  2.62818467311751 correct 44  time per epoch 0.14174485206604004
Epoch  80  loss  3.5032620840700748 correct 46  time per epoch 0.13599252700805664
Epoch  90  loss  3.3500408657510485 correct 41  time per epoch 0.1321566104888916
Epoch  100  loss  3.0022565780386206 correct 44  time per epoch 0.25861024856567383
Epoch  110  loss  0.518559010468494 correct 48  time per epoch 0.13523030281066895
Epoch  120  loss  1.2592824147263288 correct 48  time per epoch 0.1319732666015625
Epoch  130  loss  1.3019058073064942 correct 46  time per epoch 0.13648390769958496
Epoch  140  loss  2.0383950131382127 correct 48  time per epoch 0.13418149948120117
Epoch  150  loss  1.973701002391571 correct 47  time per epoch 0.13386297225952148
Epoch  160  loss  2.803564709720907 correct 48  time per epoch 0.13374853134155273
Epoch  170  loss  2.9920246767669463 correct 48  time per epoch 0.1424694061279297
Epoch  180  loss  2.442267506861585 correct 50  time per epoch 0.13746356964111328
Epoch  190  loss  1.767243137855878 correct 49  time per epoch 0.13765454292297363
Epoch  200  loss  1.0969477273362795 correct 48  time per epoch 0.13634657859802246
Epoch  210  loss  1.8070068343837393 correct 50  time per epoch 0.13381218910217285
Epoch  220  loss  2.7298417509594914 correct 48  time per epoch 0.1387791633605957
Epoch  230  loss  2.8580311166391157 correct 47  time per epoch 0.13565731048583984
Epoch  240  loss  0.7253951625543362 correct 48  time per epoch 0.13523316383361816
Epoch  250  loss  0.5221197267100426 correct 48  time per epoch 0.1495223045349121
Epoch  260  loss  0.26177621408309526 correct 50  time per epoch 0.13432788848876953
Epoch  270  loss  1.151586907325184 correct 50  time per epoch 0.2836649417877197
Epoch  280  loss  0.5631863089838803 correct 49  time per epoch 0.13832473754882812
Epoch  290  loss  1.2668389184514908 correct 50  time per epoch 0.13256478309631348
Epoch  300  loss  1.035342735868899 correct 50  time per epoch 0.1339256763458252
Epoch  310  loss  1.6797652449448628 correct 49  time per epoch 0.1407933235168457
Epoch  320  loss  0.6969096459314651 correct 49  time per epoch 0.13496899604797363
Epoch  330  loss  0.5275181601472653 correct 50  time per epoch 0.13656902313232422
Epoch  340  loss  1.3533142051869245 correct 50  time per epoch 0.13257384300231934
Epoch  350  loss  0.44469750917962214 correct 50  time per epoch 0.14216995239257812
Epoch  360  loss  1.2162952551994959 correct 50  time per epoch 0.32274794578552246
Epoch  370  loss  0.8766292692290784 correct 50  time per epoch 0.13252949714660645
Epoch  380  loss  0.7088992187296093 correct 50  time per epoch 0.13582324981689453
Epoch  390  loss  1.5026632284164294 correct 50  time per epoch 0.13334012031555176
Epoch  400  loss  1.1275006212236802 correct 50  time per epoch 0.13433337211608887
Epoch  410  loss  0.3373723304651552 correct 50  time per epoch 0.13690805435180664
Epoch  420  loss  0.653558119677541 correct 50  time per epoch 0.1423053741455078
Epoch  430  loss  0.5288466251253513 correct 50  time per epoch 0.1342172622680664
Epoch  440  loss  1.0317607195736844 correct 50  time per epoch 0.30851054191589355
Epoch  450  loss  0.6126240036332471 correct 50  time per epoch 0.13549017906188965
Epoch  460  loss  0.3970516946094922 correct 50  time per epoch 0.14458370208740234
Epoch  470  loss  0.4346125768436626 correct 50  time per epoch 0.1326000690460205
Epoch  480  loss  0.2107375963845762 correct 50  time per epoch 0.1346302032470703
Epoch  490  loss  0.8280157417896259 correct 50  time per epoch 0.13668560981750488

* Bigger model with split

Ouput training logs:

Epoch  0  loss  10.158075257479048 correct 33  time per epoch 25.417314767837524
Epoch  10  loss  4.371531914004496 correct 45  time per epoch 0.28284263610839844
Epoch  20  loss  3.7807879122964416 correct 47  time per epoch 0.611854076385498
Epoch  30  loss  4.858186458464385 correct 30  time per epoch 0.29343628883361816
Epoch  40  loss  1.7862197071219938 correct 48  time per epoch 0.284334659576416
Epoch  50  loss  1.3296859729376747 correct 47  time per epoch 0.283022403717041
Epoch  60  loss  1.6206349121690578 correct 50  time per epoch 0.5843145847320557
Epoch  70  loss  0.6322821553693282 correct 50  time per epoch 0.29588866233825684
Epoch  80  loss  1.7302682284713051 correct 49  time per epoch 0.2883296012878418
Epoch  90  loss  1.5802396458038488 correct 48  time per epoch 0.28412914276123047
Epoch  100  loss  1.5762857489013804 correct 49  time per epoch 0.37566709518432617
Epoch  110  loss  1.002481861733962 correct 50  time per epoch 0.28273463249206543
Epoch  120  loss  0.850762441496006 correct 50  time per epoch 0.28059935569763184
Epoch  130  loss  1.0671828846534064 correct 50  time per epoch 0.29842495918273926
Epoch  140  loss  0.17060321277880042 correct 50  time per epoch 0.28296589851379395
Epoch  150  loss  0.5257280562618538 correct 50  time per epoch 0.2830498218536377
Epoch  160  loss  1.9741363013767363 correct 47  time per epoch 0.2941281795501709
Epoch  170  loss  0.34200728329762253 correct 50  time per epoch 0.27730679512023926
Epoch  180  loss  0.744474683728943 correct 50  time per epoch 0.28447508811950684
Epoch  190  loss  0.26537450385177225 correct 50  time per epoch 0.283832311630249
Epoch  200  loss  0.631183467684251 correct 50  time per epoch 0.29137492179870605
Epoch  210  loss  0.45671325988068234 correct 50  time per epoch 0.281109094619751
Epoch  220  loss  0.16222176747408876 correct 50  time per epoch 0.28223490715026855
Epoch  230  loss  0.307414767288963 correct 50  time per epoch 0.2930781841278076
Epoch  240  loss  0.9112171013847208 correct 49  time per epoch 0.2851903438568115
Epoch  250  loss  0.06327185731865044 correct 50  time per epoch 0.2815368175506592
Epoch  260  loss  0.32489060950323007 correct 50  time per epoch 0.29824304580688477
Epoch  270  loss  0.17161235016619103 correct 50  time per epoch 0.2831127643585205
Epoch  280  loss  0.14404348126259237 correct 50  time per epoch 0.28260374069213867
Epoch  290  loss  0.09368660491169387 correct 50  time per epoch 0.301954984664917
Epoch  300  loss  0.08498138126502126 correct 50  time per epoch 0.2815723419189453
Epoch  310  loss  0.12424022151069337 correct 50  time per epoch 0.2796788215637207
Epoch  320  loss  0.21906946128528987 correct 50  time per epoch 0.2913949489593506
Epoch  330  loss  0.12211669913536591 correct 50  time per epoch 0.2833845615386963
Epoch  340  loss  0.32765057958518995 correct 50  time per epoch 0.28133201599121094
Epoch  350  loss  0.21825193310027038 correct 50  time per epoch 0.2870762348175049
Epoch  360  loss  0.09688651688734526 correct 50  time per epoch 0.29593420028686523
Epoch  370  loss  0.3759707107933439 correct 50  time per epoch 0.28484010696411133
Epoch  380  loss  0.49177030080392514 correct 50  time per epoch 0.2849540710449219
Epoch  390  loss  0.20694716912563513 correct 50  time per epoch 0.3972764015197754
Epoch  400  loss  0.2446776363039498 correct 50  time per epoch 0.28037118911743164
Epoch  410  loss  0.2345022979348935 correct 50  time per epoch 0.2895348072052002
Epoch  420  loss  0.13191443484160298 correct 50  time per epoch 0.29210472106933594
Epoch  430  loss  0.21211195426114612 correct 50  time per epoch 0.5954611301422119
Epoch  440  loss  0.14115856350613956 correct 50  time per epoch 0.28029894828796387
Epoch  450  loss  0.13298850104438384 correct 50  time per epoch 0.284273624420166
Epoch  460  loss  0.11507721781741304 correct 50  time per epoch 0.2872426509857178
Epoch  470  loss  0.1357379858417043 correct 50  time per epoch 0.5730717182159424
Epoch  480  loss  0.1951684329881139 correct 50  time per epoch 0.2805752754211426
Epoch  490  loss  0.08828925374131293 correct 50  time per epoch 0.2915055751800537

* Train with GPU

* Simple Dataset

Ouput training logs:

Epoch  0  loss  5.330272444007635 correct 42  time per epoch 6.045231342315674
Epoch  10  loss  1.186693814546723 correct 49  time per epoch 1.9145851135253906
Epoch  20  loss  1.731465700980256 correct 49  time per epoch 1.9430179595947266
Epoch  30  loss  0.3729199891975911 correct 49  time per epoch 2.3197476863861084
Epoch  40  loss  0.08578997379017875 correct 50  time per epoch 1.9373767375946045
Epoch  50  loss  0.6708542197094985 correct 49  time per epoch 1.912013292312622
Epoch  60  loss  0.3161332036239539 correct 50  time per epoch 2.452218770980835
Epoch  70  loss  0.43422483739753315 correct 50  time per epoch 1.9873158931732178
Epoch  80  loss  0.45508152080832254 correct 50  time per epoch 1.931687831878662
Epoch  90  loss  0.13519784323499412 correct 50  time per epoch 2.1873738765716553
Epoch  100  loss  0.5622131756070211 correct 50  time per epoch 1.9040613174438477
Epoch  110  loss  0.14761064695843776 correct 50  time per epoch 1.9277822971343994
Epoch  120  loss  0.24019584538493363 correct 50  time per epoch 2.5397074222564697
Epoch  130  loss  0.8039616019136893 correct 50  time per epoch 1.913297414779663
Epoch  140  loss  1.1255648219032612 correct 50  time per epoch 1.9355945587158203
Epoch  150  loss  0.5705476463464653 correct 50  time per epoch 2.6156744956970215
Epoch  160  loss  0.04834985869388841 correct 50  time per epoch 1.9946236610412598
Epoch  170  loss  0.06639574554891209 correct 50  time per epoch 1.9775996208190918
Epoch  180  loss  0.013490360471015055 correct 50  time per epoch 2.5576798915863037
Epoch  190  loss  0.4061816704661534 correct 50  time per epoch 1.9243276119232178
Epoch  200  loss  0.27037677815842454 correct 50  time per epoch 1.9253020286560059
Epoch  210  loss  0.10403126691846278 correct 50  time per epoch 2.374558210372925
Epoch  220  loss  0.021665718387822427 correct 50  time per epoch 1.9110503196716309
Epoch  230  loss  0.1289203707413285 correct 50  time per epoch 1.9060039520263672
Epoch  240  loss  0.009600923435702239 correct 50  time per epoch 1.9988934993743896
Epoch  250  loss  0.32658481838716213 correct 50  time per epoch 1.9862611293792725
Epoch  260  loss  0.022007307503251496 correct 50  time per epoch 1.906961441040039
Epoch  270  loss  0.47388679254488053 correct 50  time per epoch 1.9302220344543457
Epoch  280  loss  0.25569003431493137 correct 50  time per epoch 1.9161341190338135
Epoch  290  loss  0.349820171240965 correct 50  time per epoch 2.2117667198181152
Epoch  300  loss  0.009045593820967552 correct 50  time per epoch 1.9819130897521973
Epoch  310  loss  0.005573618425343562 correct 50  time per epoch 1.8973212242126465
Epoch  320  loss  0.002215510590530579 correct 50  time per epoch 2.411100387573242
Epoch  330  loss  0.005144914733659983 correct 50  time per epoch 1.9305179119110107
Epoch  340  loss  0.0745077605777727 correct 50  time per epoch 1.9874508380889893
Epoch  350  loss  0.009986862383935145 correct 50  time per epoch 2.7800419330596924
Epoch  360  loss  0.161084969592735 correct 50  time per epoch 1.9185144901275635
Epoch  370  loss  0.2743327969165576 correct 50  time per epoch 1.9024333953857422
Epoch  380  loss  0.045693108395162815 correct 50  time per epoch 2.6100966930389404
Epoch  390  loss  0.0003685962996846128 correct 50  time per epoch 1.9712204933166504
Epoch  400  loss  0.005881014627195001 correct 50  time per epoch 1.9886012077331543
Epoch  410  loss  0.020435494760506323 correct 50  time per epoch 2.6977713108062744
Epoch  420  loss  0.0012179683852549942 correct 50  time per epoch 1.9117083549499512
Epoch  430  loss  0.006286996934783385 correct 50  time per epoch 1.9203765392303467
Epoch  440  loss  0.033909767100079764 correct 50  time per epoch 2.3932695388793945
Epoch  450  loss  0.16298008056773236 correct 50  time per epoch 1.9763519763946533
Epoch  460  loss  0.1851160692884403 correct 50  time per epoch 1.898909330368042
Epoch  470  loss  0.213338413402141 correct 50  time per epoch 1.950927734375
Epoch  480  loss  0.03825541750780217 correct 50  time per epoch 1.9002470970153809
Epoch  490  loss  0.00013328604823391018 correct 50  time per epoch 2.248410224914551

* Split Dataset

Ouput training logs:

Epoch  0  loss  8.076166489787314 correct 32  time per epoch 5.22363543510437
Epoch  10  loss  6.035896179453889 correct 42  time per epoch 2.445125102996826
Epoch  20  loss  4.852683280893212 correct 47  time per epoch 1.9182467460632324
Epoch  30  loss  3.1723691535731255 correct 44  time per epoch 1.9921479225158691
Epoch  40  loss  3.08981013295883 correct 41  time per epoch 2.5485727787017822
Epoch  50  loss  5.685599359851266 correct 35  time per epoch 1.9109904766082764
Epoch  60  loss  3.796517963915198 correct 46  time per epoch 1.9244563579559326
Epoch  70  loss  3.2124677745580494 correct 43  time per epoch 2.7855989933013916
Epoch  80  loss  2.0192681429696773 correct 49  time per epoch 1.9232592582702637
Epoch  90  loss  2.282918703954904 correct 47  time per epoch 1.914536714553833
Epoch  100  loss  1.2228809775356069 correct 47  time per epoch 2.392803907394409
Epoch  110  loss  1.9951885355042305 correct 48  time per epoch 1.908949613571167
Epoch  120  loss  0.9490127445009544 correct 48  time per epoch 1.9906561374664307
Epoch  130  loss  0.7115106758657677 correct 49  time per epoch 2.555807590484619
Epoch  140  loss  0.9141257125365554 correct 47  time per epoch 1.9155542850494385
Epoch  150  loss  2.0887331665549413 correct 47  time per epoch 1.9059760570526123
Epoch  160  loss  0.17637240101083082 correct 49  time per epoch 2.779249429702759
Epoch  170  loss  1.7669060125112352 correct 48  time per epoch 1.980797529220581
Epoch  180  loss  0.3357483635694463 correct 49  time per epoch 2.1130034923553467
Epoch  190  loss  2.6777216026770962 correct 48  time per epoch 2.441148281097412
Epoch  200  loss  2.118284975448897 correct 48  time per epoch 1.921316146850586
Epoch  210  loss  1.6012926518948318 correct 48  time per epoch 1.974700927734375
Epoch  220  loss  1.0655005728917837 correct 48  time per epoch 2.6254336833953857
Epoch  230  loss  1.58823743778688 correct 48  time per epoch 1.9184646606445312
Epoch  240  loss  3.6627577482807316 correct 46  time per epoch 1.9119460582733154
Epoch  250  loss  0.5614666518499759 correct 48  time per epoch 2.784100294113159
Epoch  260  loss  0.728608719823067 correct 48  time per epoch 1.9181737899780273
Epoch  270  loss  0.5547745989795103 correct 48  time per epoch 1.9117400646209717
Epoch  280  loss  0.3284151622203429 correct 48  time per epoch 2.532118797302246
Epoch  290  loss  1.6064214002824635 correct 48  time per epoch 1.981476068496704
Epoch  300  loss  0.6195564059884184 correct 50  time per epoch 1.9860482215881348
Epoch  310  loss  2.008023369051704 correct 49  time per epoch 2.40297532081604
Epoch  320  loss  0.2592418872629249 correct 49  time per epoch 1.910494327545166
Epoch  330  loss  1.706045187669373 correct 48  time per epoch 1.9111859798431396
Epoch  340  loss  3.189699850004905 correct 46  time per epoch 2.4239869117736816
Epoch  350  loss  0.39496314376254843 correct 49  time per epoch 1.9716894626617432
Epoch  360  loss  1.6727945246337639 correct 47  time per epoch 1.935925006866455
Epoch  370  loss  0.9718984374197036 correct 50  time per epoch 2.225078582763672
Epoch  380  loss  0.18875666080102416 correct 50  time per epoch 1.9456782341003418
Epoch  390  loss  1.049463465206761 correct 50  time per epoch 1.9758620262145996
Epoch  400  loss  0.19635197488032474 correct 50  time per epoch 2.715167284011841
Epoch  410  loss  0.08083717589701224 correct 49  time per epoch 1.891829490661621
Epoch  420  loss  0.3967749336894624 correct 49  time per epoch 1.9230802059173584
Epoch  430  loss  0.33386140636091693 correct 50  time per epoch 2.3986129760742188
Epoch  440  loss  1.7827692817756275 correct 47  time per epoch 2.0307934284210205
Epoch  450  loss  0.27722112229940493 correct 48  time per epoch 2.0038914680480957
Epoch  460  loss  0.3240325669373439 correct 49  time per epoch 2.249403953552246
Epoch  470  loss  0.7968554649930322 correct 50  time per epoch 1.9717190265655518
Epoch  480  loss  0.45737838667299885 correct 50  time per epoch 1.9470977783203125
Epoch  490  loss  0.40247267723437063 correct 50  time per epoch 2.2476532459259033


* Xor Dataset

Ouput training logs:

Epoch  0  loss  5.730375032252167 correct 36  time per epoch 4.392451286315918
Epoch  10  loss  4.033176488094769 correct 38  time per epoch 2.1971709728240967
Epoch  20  loss  3.992683895134078 correct 44  time per epoch 1.9353551864624023
Epoch  30  loss  2.7948544357929057 correct 45  time per epoch 1.9856913089752197
Epoch  40  loss  1.4966150270409715 correct 45  time per epoch 2.4240100383758545
Epoch  50  loss  5.035889332311076 correct 47  time per epoch 1.9338953495025635
Epoch  60  loss  2.445232266049869 correct 49  time per epoch 1.9264154434204102
Epoch  70  loss  3.1474686796983264 correct 46  time per epoch 2.7516415119171143
Epoch  80  loss  3.846994548912881 correct 48  time per epoch 1.9261515140533447
Epoch  90  loss  1.7691823792707262 correct 48  time per epoch 1.9144916534423828
Epoch  100  loss  1.4204964463548948 correct 47  time per epoch 2.254664659500122
Epoch  110  loss  2.015831099790105 correct 44  time per epoch 1.893876314163208
Epoch  120  loss  1.1826715521470905 correct 46  time per epoch 1.9683804512023926
Epoch  130  loss  3.2255089602801226 correct 48  time per epoch 1.8904309272766113
Epoch  140  loss  2.0022671947143396 correct 47  time per epoch 1.895845890045166
Epoch  150  loss  1.4831536328416683 correct 48  time per epoch 2.00110125541687
Epoch  160  loss  1.336629707586651 correct 47  time per epoch 1.983766794204712
Epoch  170  loss  0.6941357524806041 correct 49  time per epoch 1.9657361507415771
Epoch  180  loss  0.7046015816357653 correct 49  time per epoch 2.4312052726745605
Epoch  190  loss  2.5392833988043235 correct 49  time per epoch 1.8956577777862549
Epoch  200  loss  2.031509928203098 correct 48  time per epoch 1.8962159156799316
Epoch  210  loss  0.963821581097749 correct 49  time per epoch 2.7431390285491943
Epoch  220  loss  1.8600295171308632 correct 49  time per epoch 1.9070839881896973
Epoch  230  loss  0.4085271868734231 correct 49  time per epoch 1.8995649814605713
Epoch  240  loss  2.9270253907292654 correct 48  time per epoch 2.389200448989868
Epoch  250  loss  0.719464150956681 correct 49  time per epoch 1.9733364582061768
Epoch  260  loss  2.1718708865523886 correct 48  time per epoch 1.8748812675476074
Epoch  270  loss  0.7760881104215066 correct 49  time per epoch 1.9576990604400635
Epoch  280  loss  2.380521343716483 correct 49  time per epoch 1.9275314807891846
Epoch  290  loss  0.761948132747963 correct 49  time per epoch 2.013333797454834
Epoch  300  loss  2.316922178391144 correct 49  time per epoch 1.9654719829559326
Epoch  310  loss  0.44961633833379544 correct 49  time per epoch 1.9082419872283936
Epoch  320  loss  1.084690003513397 correct 49  time per epoch 1.9737355709075928
Epoch  330  loss  0.39907993792586516 correct 49  time per epoch 1.8990068435668945
Epoch  340  loss  0.4497947077328826 correct 49  time per epoch 1.976210594177246
Epoch  350  loss  1.49404668997884 correct 49  time per epoch 2.482710838317871
Epoch  360  loss  0.8221548806364709 correct 49  time per epoch 1.8928823471069336
Epoch  370  loss  0.5146957714714693 correct 48  time per epoch 1.893120288848877
Epoch  380  loss  0.5498876965028634 correct 49  time per epoch 2.7084121704101562
Epoch  390  loss  2.185779660823633 correct 48  time per epoch 1.9632189273834229
Epoch  400  loss  1.3522875579298597 correct 49  time per epoch 1.9620287418365479
Epoch  410  loss  0.25645818098093354 correct 49  time per epoch 2.1427268981933594
Epoch  420  loss  0.30096683284898174 correct 49  time per epoch 1.8940117359161377
Epoch  430  loss  0.2954325967801207 correct 49  time per epoch 1.9163756370544434
Epoch  440  loss  1.2425445974865434 correct 49  time per epoch 2.1906344890594482
Epoch  450  loss  0.29311572398977287 correct 48  time per epoch 1.9639358520507812
Epoch  460  loss  1.1590091629148425 correct 49  time per epoch 1.8940322399139404
Epoch  470  loss  1.7045521435644104 correct 49  time per epoch 1.9015076160430908
Epoch  480  loss  2.0816517750704437 correct 50  time per epoch 1.8960039615631104
Epoch  490  loss  0.1459161930566848 correct 50  time per epoch 2.2729403972625732

* Bigger model with split

Ouput training logs:

Epoch  0  loss  7.748797447100827 correct 26  time per epoch 4.389522314071655
Epoch  10  loss  5.156631428423642 correct 37  time per epoch 2.198568344116211
Epoch  20  loss  3.156837556781007 correct 45  time per epoch 2.016918659210205
Epoch  30  loss  3.7035675579661134 correct 38  time per epoch 2.0562918186187744
Epoch  40  loss  1.1952938490061322 correct 43  time per epoch 2.0055744647979736
Epoch  50  loss  4.29470710612398 correct 45  time per epoch 2.5354464054107666
Epoch  60  loss  2.9438079203178207 correct 49  time per epoch 2.0199081897735596
Epoch  70  loss  1.5613671596654561 correct 48  time per epoch 2.055203914642334
Epoch  80  loss  4.538248683987109 correct 41  time per epoch 1.9809088706970215
Epoch  90  loss  1.5044213718688544 correct 46  time per epoch 2.4094626903533936
Epoch  100  loss  3.1536320695252513 correct 44  time per epoch 1.9831960201263428
Epoch  110  loss  5.101712594008287 correct 42  time per epoch 2.0048670768737793
Epoch  120  loss  2.5468043794024275 correct 45  time per epoch 2.530778169631958
Epoch  130  loss  2.629571312965714 correct 45  time per epoch 2.0053186416625977
Epoch  140  loss  0.7582214298051277 correct 47  time per epoch 1.9908421039581299
Epoch  150  loss  0.5925129279397385 correct 47  time per epoch 1.987891435623169
Epoch  160  loss  1.594468318904907 correct 47  time per epoch 2.4214255809783936
Epoch  170  loss  2.8968343444241507 correct 45  time per epoch 2.0579421520233154
Epoch  180  loss  2.8225446964543006 correct 49  time per epoch 1.9963934421539307
Epoch  190  loss  6.093696861461916 correct 43  time per epoch 2.614292860031128
Epoch  200  loss  1.087579666185243 correct 46  time per epoch 1.9991049766540527
Epoch  210  loss  2.1792391151410593 correct 49  time per epoch 2.070584297180176
Epoch  220  loss  2.7313712652710556 correct 46  time per epoch 2.0109589099884033
Epoch  230  loss  3.761595287113505 correct 45  time per epoch 2.2204840183258057
Epoch  240  loss  1.028636702264048 correct 46  time per epoch 2.0785629749298096
Epoch  250  loss  0.8485668877923 correct 48  time per epoch 2.03582501411438
Epoch  260  loss  3.0412225756843694 correct 47  time per epoch 2.481168031692505
Epoch  270  loss  1.6794291591423833 correct 48  time per epoch 1.9850282669067383
Epoch  280  loss  0.7748574542049792 correct 47  time per epoch 1.9712426662445068
Epoch  290  loss  2.879964902342668 correct 45  time per epoch 2.073502540588379
Epoch  300  loss  1.993504651367442 correct 46  time per epoch 2.516195774078369
Epoch  310  loss  0.9199778887635668 correct 46  time per epoch 2.014643669128418
Epoch  320  loss  1.2714025673071656 correct 47  time per epoch 2.0089809894561768
Epoch  330  loss  0.7421452907213493 correct 47  time per epoch 2.538583993911743
Epoch  340  loss  1.914488885969389 correct 47  time per epoch 2.0472230911254883
Epoch  350  loss  2.114845191914699 correct 45  time per epoch 2.082303285598755
Epoch  360  loss  0.792253703750347 correct 48  time per epoch 1.9902684688568115
Epoch  370  loss  2.057192701688309 correct 49  time per epoch 2.792356014251709
Epoch  380  loss  0.3661997346663801 correct 47  time per epoch 1.9834568500518799
Epoch  390  loss  1.7556554368026007 correct 45  time per epoch 2.0487451553344727
Epoch  400  loss  0.6334077871556386 correct 48  time per epoch 2.251498222351074
Epoch  410  loss  0.8905263618551335 correct 50  time per epoch 2.051440954208374
Epoch  420  loss  0.918993575949511 correct 46  time per epoch 1.9817421436309814
Epoch  430  loss  0.2787654458853847 correct 48  time per epoch 1.9808118343353271
Epoch  440  loss  3.2432877050950273 correct 48  time per epoch 2.809041976928711
Epoch  450  loss  0.13678956707107592 correct 49  time per epoch 2.0700643062591553
Epoch  460  loss  1.30038880622927 correct 49  time per epoch 1.9972286224365234
Epoch  470  loss  1.0050865347991522 correct 49  time per epoch 2.5625362396240234
Epoch  480  loss  0.8484477080385332 correct 50  time per epoch 1.9709067344665527
Epoch  490  loss  1.497741865351589 correct 50  time per epoch 2.040520668029785
