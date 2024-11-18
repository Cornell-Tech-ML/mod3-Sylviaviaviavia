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

Epoch  0  loss  7.316796363494368 correct 42
Epoch  10  loss  3.9918724187095638 correct 44
Epoch  20  loss  2.2514783673133802 correct 44
Epoch  30  loss  1.6143498998303312 correct 46
Epoch  40  loss  0.7475266079932423 correct 46
Epoch  50  loss  1.3375145500987788 correct 50
Epoch  60  loss  1.196743632880092 correct 50
Epoch  70  loss  1.4440699750494144 correct 50
Epoch  80  loss  0.6528424969903592 correct 50
Epoch  90  loss  1.8440933998019169 correct 48
Epoch  100  loss  0.9278881019767651 correct 50
Epoch  110  loss  1.47682887535186 correct 50
Epoch  120  loss  0.8923862893647306 correct 50
Epoch  130  loss  1.1174240968293705 correct 50
Epoch  140  loss  1.0657975159315451 correct 50
Epoch  150  loss  0.49543703326769545 correct 50
Epoch  160  loss  0.3999729152802294 correct 50
Epoch  170  loss  0.6790113235153856 correct 50
Epoch  180  loss  0.4744895627285353 correct 50
Epoch  190  loss  0.48831092515072927 correct 50
Epoch  200  loss  0.47064893122465445 correct 50
Epoch  210  loss  0.1559184098691941 correct 50
Epoch  220  loss  0.3272877431690036 correct 50
Epoch  230  loss  0.43174300264083254 correct 50
Epoch  240  loss  0.5105293148173721 correct 50
Epoch  250  loss  0.23320517832555238 correct 50
Epoch  260  loss  0.19936463563756596 correct 50
Epoch  270  loss  0.6542777273857856 correct 50
Epoch  280  loss  0.09330294108652275 correct 50
Epoch  290  loss  0.36550503930754547 correct 50
Epoch  300  loss  0.22391384354874294 correct 50
Epoch  310  loss  0.2453217809327048 correct 50
Epoch  320  loss  0.2173195446782544 correct 50
Epoch  330  loss  0.3305406793088237 correct 50
Epoch  340  loss  0.18227324497894346 correct 50
Epoch  350  loss  0.132011821122147 correct 50
Epoch  360  loss  0.17400682361530564 correct 50
Epoch  370  loss  0.1915685634560433 correct 50
Epoch  380  loss  0.12604618589387243 correct 50
Epoch  390  loss  0.14939729161410054 correct 50
Epoch  400  loss  0.2501674629840213 correct 50
Epoch  410  loss  0.19395066399990632 correct 50
Epoch  420  loss  0.2824469086797419 correct 50
Epoch  430  loss  0.10399695695775708 correct 50
Epoch  440  loss  0.1427464221066188 correct 50
Epoch  450  loss  0.05842940787607116 correct 50
Epoch  460  loss  0.17389390202081834 correct 50
Epoch  470  loss  0.13672259671542747 correct 50
Epoch  480  loss  0.19516350361507265 correct 50
Epoch  490  loss  0.14614726335466158 correct 50


* Split Dataset

Ouput training logs:

Epoch  0  loss  6.388781577917608 correct 39
Epoch  10  loss  5.581347286785615 correct 39
Epoch  20  loss  4.151578197519736 correct 46
Epoch  30  loss  3.9078239804298263 correct 41
Epoch  40  loss  3.6162431148797065 correct 37
Epoch  50  loss  4.11150313884348 correct 48
Epoch  60  loss  2.2236540548330095 correct 45
Epoch  70  loss  3.0461227005971887 correct 49
Epoch  80  loss  2.255614312730671 correct 47
Epoch  90  loss  3.3520815764244167 correct 46
Epoch  100  loss  1.9486061476667373 correct 50
Epoch  110  loss  2.5952915315775718 correct 47
Epoch  120  loss  1.51319534315571 correct 48
Epoch  130  loss  1.3688135636190557 correct 49
Epoch  140  loss  1.0292506738452616 correct 49
Epoch  150  loss  1.1745018911359921 correct 50
Epoch  160  loss  0.2552056293560028 correct 49
Epoch  170  loss  2.4059985193550832 correct 49
Epoch  180  loss  0.5711320485647184 correct 49
Epoch  190  loss  1.5061488604588575 correct 49
Epoch  200  loss  0.8584505546206643 correct 49
Epoch  210  loss  0.6086435211320405 correct 49
Epoch  220  loss  1.0275464993741241 correct 50
Epoch  230  loss  1.3233727946157876 correct 50
Epoch  240  loss  1.5697861291510797 correct 50
Epoch  250  loss  0.7050503229335651 correct 50
Epoch  260  loss  0.4933354425352426 correct 50
Epoch  270  loss  1.1028134267557408 correct 50
Epoch  280  loss  0.8022229664705357 correct 50
Epoch  290  loss  0.6120724384242285 correct 50
Epoch  300  loss  0.04080204571593383 correct 50
Epoch  310  loss  0.4127944264343665 correct 50
Epoch  320  loss  0.181116679832378 correct 50
Epoch  330  loss  0.5155123508784567 correct 49
Epoch  340  loss  0.16636087522116522 correct 50
Epoch  350  loss  0.2163180828803616 correct 49
Epoch  360  loss  1.1607830338412848 correct 50
Epoch  370  loss  0.7032737512036282 correct 49
Epoch  380  loss  0.9454962402987714 correct 50
Epoch  390  loss  0.6678050390615364 correct 49
Epoch  400  loss  0.09197502562698717 correct 50
Epoch  410  loss  0.49666048056183076 correct 50
Epoch  420  loss  0.28987482877527815 correct 50
Epoch  430  loss  0.08535550194927853 correct 50
Epoch  440  loss  0.480653863818886 correct 49
Epoch  450  loss  0.9824052495143293 correct 50
Epoch  460  loss  0.24926776555017394 correct 50
Epoch  470  loss  0.23643927164837958 correct 50
Epoch  480  loss  0.40950366432608365 correct 50
Epoch  490  loss  0.7136261621410799 correct 49


* Xor Dataset

Ouput training logs:

Epoch  0  loss  7.454349529417256 correct 32
Epoch  10  loss  6.020958305354137 correct 46
Epoch  20  loss  4.139935831634276 correct 43
Epoch  30  loss  2.8003288464820466 correct 46
Epoch  40  loss  4.076030070669775 correct 48
Epoch  50  loss  3.0273612097888507 correct 46
Epoch  60  loss  2.77542884603071 correct 48
Epoch  70  loss  3.8028060339242717 correct 48
Epoch  80  loss  1.1757209402534465 correct 48
Epoch  90  loss  2.1703813841723845 correct 49
Epoch  100  loss  1.2766479538361164 correct 49
Epoch  110  loss  1.1822098934039182 correct 49
Epoch  120  loss  2.3502134903674463 correct 48
Epoch  130  loss  0.5836776174243752 correct 48
Epoch  140  loss  0.47456155948616474 correct 49
Epoch  150  loss  0.7196098428726182 correct 49
Epoch  160  loss  1.8511012402187697 correct 48
Epoch  170  loss  1.5008741171854982 correct 48
Epoch  180  loss  0.9536583331024849 correct 49
Epoch  190  loss  1.0958077847592773 correct 49
Epoch  200  loss  1.0870583855116531 correct 49
Epoch  210  loss  1.8219072126501783 correct 49
Epoch  220  loss  0.3733454579927162 correct 49
Epoch  230  loss  1.7314467179957878 correct 49
Epoch  240  loss  0.7236988424362082 correct 49
Epoch  250  loss  0.328177323059125 correct 49
Epoch  260  loss  0.19072442039703244 correct 50
Epoch  270  loss  0.6201269415677322 correct 49
Epoch  280  loss  0.6819164820195485 correct 49
Epoch  290  loss  2.7127810689141945 correct 49
Epoch  300  loss  1.9439667290379012 correct 49
Epoch  310  loss  1.5333566492693673 correct 49
Epoch  320  loss  0.9858757776273136 correct 49
Epoch  330  loss  0.23008543457517655 correct 49
Epoch  340  loss  0.812062677208767 correct 50
Epoch  350  loss  0.8709329943292468 correct 50
Epoch  360  loss  0.34952415690779554 correct 49
Epoch  370  loss  0.35910473277275806 correct 49
Epoch  380  loss  1.4249188327364235 correct 49
Epoch  390  loss  0.9866284554356881 correct 49
Epoch  400  loss  0.5500248718871322 correct 49
Epoch  410  loss  0.2718085769345858 correct 49
Epoch  420  loss  1.0281986433935666 correct 49
Epoch  430  loss  0.22224254771975396 correct 49
Epoch  440  loss  0.39213761538570524 correct 50
Epoch  450  loss  0.6233416607640855 correct 50
Epoch  460  loss  1.9429096915453004 correct 49
Epoch  470  loss  0.061271038914325365 correct 49
Epoch  480  loss  0.0918965411736389 correct 50
Epoch  490  loss  0.04310904751515741 correct 50

* Bigger model with split

Ouput training logs:

Epoch  0  loss  7.316796363494368 correct 42
Epoch  10  loss  3.9918724187095638 correct 44
Epoch  20  loss  2.2514783673133802 correct 44
Epoch  30  loss  1.6143498998303312 correct 46
Epoch  40  loss  0.7475266079932423 correct 46
Epoch  50  loss  1.3375145500987788 correct 50
Epoch  60  loss  1.196743632880092 correct 50
Epoch  70  loss  1.4440699750494144 correct 50
Epoch  80  loss  0.6528424969903592 correct 50
Epoch  90  loss  1.8440933998019169 correct 48
Epoch  100  loss  0.9278881019767651 correct 50
Epoch  110  loss  1.47682887535186 correct 50
Epoch  120  loss  0.8923862893647306 correct 50
Epoch  130  loss  1.1174240968293705 correct 50
Epoch  140  loss  1.0657975159315451 correct 50
Epoch  150  loss  0.49543703326769545 correct 50
Epoch  160  loss  0.3999729152802294 correct 50
Epoch  170  loss  0.6790113235153856 correct 50
Epoch  180  loss  0.4744895627285353 correct 50
Epoch  190  loss  0.48831092515072927 correct 50
Epoch  200  loss  0.47064893122465445 correct 50
Epoch  210  loss  0.1559184098691941 correct 50
Epoch  220  loss  0.3272877431690036 correct 50
Epoch  230  loss  0.43174300264083254 correct 50
Epoch  240  loss  0.5105293148173721 correct 50
Epoch  250  loss  0.23320517832555238 correct 50
Epoch  260  loss  0.19936463563756596 correct 50
Epoch  270  loss  0.6542777273857856 correct 50
Epoch  280  loss  0.09330294108652275 correct 50
Epoch  290  loss  0.36550503930754547 correct 50
Epoch  300  loss  0.22391384354874294 correct 50
Epoch  310  loss  0.2453217809327048 correct 50
Epoch  320  loss  0.2173195446782544 correct 50
Epoch  330  loss  0.3305406793088237 correct 50
Epoch  340  loss  0.18227324497894346 correct 50
Epoch  350  loss  0.132011821122147 correct 50
Epoch  360  loss  0.17400682361530564 correct 50
Epoch  370  loss  0.1915685634560433 correct 50
Epoch  380  loss  0.12604618589387243 correct 50
Epoch  390  loss  0.14939729161410054 correct 50
Epoch  400  loss  0.2501674629840213 correct 50
Epoch  410  loss  0.19395066399990632 correct 50
Epoch  420  loss  0.2824469086797419 correct 50
Epoch  430  loss  0.10399695695775708 correct 50
Epoch  440  loss  0.1427464221066188 correct 50
Epoch  450  loss  0.05842940787607116 correct 50
Epoch  460  loss  0.17389390202081834 correct 50
Epoch  470  loss  0.13672259671542747 correct 50
Epoch  480  loss  0.19516350361507265 correct 50
Epoch  490  loss  0.14614726335466158 correct 50

* Train with GPU

* Simple Dataset

Ouput training logs:

Epoch  0  loss  7.231053993513349 correct 22
Epoch  10  loss  5.592493000865293 correct 39
Epoch  20  loss  3.6894614562866073 correct 42
Epoch  30  loss  3.4491366425815793 correct 45
Epoch  40  loss  3.7876306097372603 correct 47
Epoch  50  loss  2.9216280443057348 correct 49
Epoch  60  loss  3.465394643322442 correct 49
Epoch  70  loss  2.036619168224319 correct 49
Epoch  80  loss  2.7896875446032796 correct 49
Epoch  90  loss  1.645696669833359 correct 47
Epoch  100  loss  1.699465237035873 correct 50
Epoch  110  loss  1.269946397964325 correct 50
Epoch  120  loss  1.0284938948895337 correct 50
Epoch  130  loss  0.9671019719531018 correct 46
Epoch  140  loss  3.077931399556466 correct 48
Epoch  150  loss  1.5008067390834987 correct 50
Epoch  160  loss  1.0387122363515073 correct 49
Epoch  170  loss  0.8613384530380962 correct 50
Epoch  180  loss  1.680688331336344 correct 49
Epoch  190  loss  0.24124398360264404 correct 50
Epoch  200  loss  1.4569043328284565 correct 50
Epoch  210  loss  0.6000568675740479 correct 49
Epoch  220  loss  0.6967216028550631 correct 49
Epoch  230  loss  0.3374720867176588 correct 50
Epoch  240  loss  0.7945311122166494 correct 50
Epoch  250  loss  0.12066176869386604 correct 50
Epoch  260  loss  0.28848629771884654 correct 50
Epoch  270  loss  0.07760392830527571 correct 50
Epoch  280  loss  1.0526480377520109 correct 50
Epoch  290  loss  0.1478037730336977 correct 49
Epoch  300  loss  0.309037652000728 correct 50
Epoch  310  loss  0.9603883602753455 correct 50
Epoch  320  loss  0.7433593219082215 correct 49
Epoch  330  loss  0.572709047219741 correct 50
Epoch  340  loss  1.3498409305961248 correct 50
Epoch  350  loss  0.7063558879011146 correct 50
Epoch  360  loss  0.4215000954921694 correct 50
Epoch  370  loss  0.2190462940383191 correct 50
Epoch  380  loss  0.15047673493481512 correct 50
Epoch  390  loss  1.4388936726657333 correct 49
Epoch  400  loss  0.35311529211205844 correct 50
Epoch  410  loss  0.08009747850962115 correct 50
Epoch  420  loss  0.4047619472379198 correct 50
Epoch  430  loss  0.4827203254842322 correct 50
Epoch  440  loss  0.4138447015744856 correct 50
Epoch  450  loss  0.11762447947433768 correct 50
Epoch  460  loss  0.13407218154819578 correct 50
Epoch  470  loss  0.033674130675310714 correct 50
Epoch  480  loss  0.06390818862576311 correct 50
Epoch  490  loss  0.5293586179206253 correct 50

* Split Dataset

Ouput training logs:

Epoch  0  loss  5.567673835588447 correct 34
Epoch  10  loss  4.842030209828455 correct 42
Epoch  20  loss  7.2085311916487775 correct 46
Epoch  30  loss  2.519241439575286 correct 46
Epoch  40  loss  3.3127626360471263 correct 49
Epoch  50  loss  3.0180620344232145 correct 49
Epoch  60  loss  1.9610898064043507 correct 48
Epoch  70  loss  2.0497073731540274 correct 48
Epoch  80  loss  0.9969673513081753 correct 49
Epoch  90  loss  1.054668000954052 correct 49
Epoch  100  loss  1.0466268271064256 correct 49
Epoch  110  loss  0.6006120190220481 correct 50
Epoch  120  loss  0.6694357143374453 correct 50
Epoch  130  loss  0.9038997672727893 correct 50
Epoch  140  loss  0.38087866189863956 correct 50
Epoch  150  loss  0.6417929206091052 correct 49
Epoch  160  loss  0.4246649345986972 correct 50
Epoch  170  loss  0.5559809108026289 correct 50
Epoch  180  loss  0.15939371127351543 correct 50
Epoch  190  loss  0.5806843771089992 correct 49
Epoch  200  loss  0.4793410172701272 correct 50
Epoch  210  loss  0.8774630934276919 correct 50
Epoch  220  loss  0.5496315133349859 correct 50
Epoch  230  loss  0.4675805140451428 correct 50
Epoch  240  loss  0.1522432652133706 correct 50
Epoch  250  loss  0.9746612480241397 correct 50
Epoch  260  loss  0.21407091571103234 correct 50
Epoch  270  loss  0.5091741629588534 correct 50
Epoch  280  loss  0.7647754661533748 correct 50
Epoch  290  loss  0.051064867313541755 correct 50
Epoch  300  loss  0.49709362347848385 correct 50
Epoch  310  loss  0.7876838617512835 correct 50
Epoch  320  loss  0.6475966081038084 correct 50
Epoch  330  loss  0.504164672795862 correct 50
Epoch  340  loss  0.34919498923595826 correct 50
Epoch  350  loss  0.40412798032296393 correct 50
Epoch  360  loss  0.28884764531278195 correct 50
Epoch  370  loss  0.5574152932501755 correct 50
Epoch  380  loss  0.3325730272619212 correct 50
Epoch  390  loss  0.47758528999781297 correct 50
Epoch  400  loss  0.4960425537323784 correct 50
Epoch  410  loss  0.15918748149867679 correct 50
Epoch  420  loss  0.4602667617007924 correct 50
Epoch  430  loss  0.23894722841184443 correct 50
Epoch  440  loss  0.1528521636781177 correct 50
Epoch  450  loss  0.32238023913832914 correct 50
Epoch  460  loss  0.41578678388308676 correct 50
Epoch  470  loss  0.05372885626901983 correct 50
Epoch  480  loss  0.3013546414448517 correct 50
Epoch  490  loss  0.11879520944154881 correct 50



* Xor Dataset

Ouput training logs:

Epoch  0  loss  6.645000896939344 correct 33
Epoch  10  loss  3.9895462631630143 correct 40
Epoch  20  loss  4.559656461358913 correct 41
Epoch  30  loss  4.69842618437298 correct 43
Epoch  40  loss  3.7038761148936534 correct 45
Epoch  50  loss  2.2130523649296268 correct 45
Epoch  60  loss  2.307398808699361 correct 45
Epoch  70  loss  4.076705184605394 correct 44
Epoch  80  loss  3.479062782575043 correct 45
Epoch  90  loss  1.4819280137893522 correct 45
Epoch  100  loss  2.541124648210621 correct 46
Epoch  110  loss  2.7342649263733416 correct 46
Epoch  120  loss  3.694558358418848 correct 47
Epoch  130  loss  2.070719498207743 correct 46
Epoch  140  loss  2.5907035760055983 correct 49
Epoch  150  loss  1.9445040050940627 correct 47
Epoch  160  loss  1.4167527086328755 correct 47
Epoch  170  loss  1.2838008347353236 correct 48
Epoch  180  loss  0.9962644663624272 correct 48
Epoch  190  loss  0.7465855118225184 correct 49
Epoch  200  loss  2.2576381194185475 correct 49
Epoch  210  loss  1.6918826723026883 correct 49
Epoch  220  loss  1.4363150635360429 correct 50
Epoch  230  loss  1.1846720935134878 correct 50
Epoch  240  loss  0.991050397823865 correct 50
Epoch  250  loss  1.4690613679621842 correct 50
Epoch  260  loss  1.1024337660900647 correct 50
Epoch  270  loss  0.9114310265973987 correct 50
Epoch  280  loss  0.9026795608886875 correct 50
Epoch  290  loss  0.6345582862234104 correct 50
Epoch  300  loss  1.1870432475575574 correct 50
Epoch  310  loss  0.6895284645897668 correct 50
Epoch  320  loss  0.7724705879618586 correct 50
Epoch  330  loss  0.40814404752966305 correct 50
Epoch  340  loss  0.300301420379702 correct 50
Epoch  350  loss  0.5860859011358356 correct 50
Epoch  360  loss  0.49190715604712343 correct 50
Epoch  370  loss  0.9416243744290512 correct 50
Epoch  380  loss  0.7225658920160891 correct 50
Epoch  390  loss  0.5111091130189912 correct 50
Epoch  400  loss  0.16042663916336863 correct 50
Epoch  410  loss  1.0371018304233894 correct 50
Epoch  420  loss  0.883827030724571 correct 50
Epoch  430  loss  0.41753439552203003 correct 50
Epoch  440  loss  0.6744919873343023 correct 50
Epoch  450  loss  0.06018560216850463 correct 50
Epoch  460  loss  0.26205859014992594 correct 50
Epoch  470  loss  0.1553538119752879 correct 50
Epoch  480  loss  0.16916837915738525 correct 50
Epoch  490  loss  0.05875481749727439 correct 50

* Bigger model with split

Ouput training logs:

Epoch  0  loss  21.201100034634496 correct 34
Epoch  10  loss  5.537843431334952 correct 36
Epoch  20  loss  4.779238942053505 correct 38
Epoch  30  loss  5.001201038286015 correct 47
Epoch  40  loss  1.1274368267000852 correct 45
Epoch  50  loss  2.784554150451369 correct 47
Epoch  60  loss  1.9701137241046047 correct 47
Epoch  70  loss  1.7660645004115112 correct 50
Epoch  80  loss  2.252075653833027 correct 50
Epoch  90  loss  1.3929282171197068 correct 50
Epoch  100  loss  0.21694469883395617 correct 49
Epoch  110  loss  1.414592230851867 correct 50
Epoch  120  loss  1.0919347521835459 correct 50
Epoch  130  loss  1.3971849459331385 correct 50
Epoch  140  loss  0.3987947048246837 correct 50
Epoch  150  loss  0.9056887193357899 correct 50
Epoch  160  loss  0.7286242601344457 correct 50
Epoch  170  loss  0.9694945565713875 correct 50
Epoch  180  loss  1.178831505843255 correct 48
Epoch  190  loss  0.10981495640020449 correct 50
Epoch  200  loss  0.2838538505748787 correct 50
Epoch  210  loss  0.4383391825393563 correct 50
Epoch  220  loss  0.3854908995314356 correct 50
Epoch  230  loss  0.28970344272440596 correct 50
Epoch  240  loss  0.3728194302974576 correct 50
Epoch  250  loss  0.3845574073244003 correct 50
Epoch  260  loss  0.3950945140255365 correct 50
Epoch  270  loss  0.1289480932641517 correct 50
Epoch  280  loss  0.7480821311431294 correct 50
Epoch  290  loss  0.19618174128616253 correct 50
Epoch  300  loss  0.377521528150789 correct 50
Epoch  310  loss  0.20834298777434004 correct 50
Epoch  320  loss  0.07763046301023854 correct 50
Epoch  330  loss  0.2602048644149415 correct 50
Epoch  340  loss  0.25681132348207997 correct 50
Epoch  350  loss  0.1672231424472574 correct 50
Epoch  360  loss  0.04852856559620498 correct 50
Epoch  370  loss  0.08438126818271768 correct 50
Epoch  380  loss  0.2365306507989156 correct 50
Epoch  390  loss  0.3127582293824695 correct 50
Epoch  400  loss  0.282264740143272 correct 50
Epoch  410  loss  0.3859944511292444 correct 50
Epoch  420  loss  0.29238053020439825 correct 50
Epoch  430  loss  0.00994596132589869 correct 50
Epoch  440  loss  0.09847407919147903 correct 50
Epoch  450  loss  0.35067546186219933 correct 50
Epoch  460  loss  0.18175864218110022 correct 50
Epoch  470  loss  0.18599894955743784 correct 50
Epoch  480  loss  0.1504590548425493 correct 50
Epoch  490  loss  0.18556766040423175 correct 50