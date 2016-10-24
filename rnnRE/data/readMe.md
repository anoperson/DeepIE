This is the dataset used in the following paper:

Combining Neural Networks and Log-linear Models to Improve Relation Extraction
Thien Huu Nguyen and Ralph Grishman, in Proceedings of IJCAI Workshop on Deep Learning for Artificial Intelligence, New York, USA, 2016.

######
There are 15 data files (.txt) in this directory.

In the paper, our experiments are done with:

+ bn_nw.full.txt for training
+ bc0.full.txt for development
+ bc1.full.txt, cts.full.txt, wl.full.txt as test data on different domain

This release also supports domain adaptation experiments where the out-out-domain experiments are done with the similar files presented above. In order to obtain the in-domain performance (with 5-fold crossvalidation, train your systems on bn_nw_train$1.full.txt and evaluate it on bn_nw_test$i.full.txt ($i = 0,1,2,3,4) using bc0.full.txt for development (union of bn_nw_test$i.full.txt is bn_nw.full.txt)

####
The format of the files in dataset is as follow.

Each line in a file corresponds to an example (relation mention) for relation extraction. The lines should be separted by tab "\t" with the meaning as follow:

Fields 0: id for the relation mention
Field 1: golden type of the relation mention (*considering the directionality of the relation mentions*)
Field 2: string for the smallest subtree of the constituent tree covering the two head words: readily used for the kernel methods with svm
Field 3: format: typeOfEntityMention1@typeOfEntityMention2 (e.g, GPE@ORG, PER@PER ...)
Field 4: traditional binary features extracted for this entity mention (should be splitted by spaces and ignore the first field qid:*)
Field 5: sentence containing this relation mention (already tokenized). Each token is associated with the BIO entty mention annotation (i.e, B-PER, I-PER, O ...) and entity mention type (i.e, NAM, PRO ...). Tokens of the two entity mention of interest are further tagged with numbers "1" and "2" to indicate the first and the second entity mentions. An example of filed 4 looks like:

I/O want/O to/O join/O together/O the/O feelings/O of/O each/O of/O us/B-PER.PRO#1 as/O individuals/O who/B-PER.PRO#2 oppose/O the/O war/O ./O

Note that if there are multiple tokens for a single entity mention (tagged with the same number "1" or "2"), we simply use the last token as the position for the entity mention.

Field 6: for every token in the sentence, we compute the least common parent nodes with the two last tokens of the two entity mentions in the constituent tree (so there would be two nodes in the tree for each token in the sentence). We take the labels of the nodes as features for the current token. This field should be separated by spaces, each resulting element correponds to a token in the sentence and should be separated by "--" to obtain the two node labels.
Field 7: part of speech sequence for the tokens in the sentence (splitted by spaces, same length with filed 4 when splitted by spaces)
Field 8: BIO chunking annotation sequence for the tokens in the sentence
Field 9: dependency path between two entity mentions of interest (using the last tokens in each entity mention as anchors, should be splitted by spaces)
Field 10: sequence of dependency relations surrounding the tokens in the dependency tree (separted by "@" and then each resulting element should be separated by spaces)
Filed 11: sequence of governor words for the tokens in the dependency tree (splitted by spaces)
Field 12: relation tripples of the dependency tree for the current sentence (separated by "######" to get the tripples) (so you don't need to run a parser again to get the trees)

Note that for bn_nw_test*.full.txt and bn_nw_train*.full.txt, there are no strings for the smallest constituent tree (field 2) above, and the binary features (field 4 above) is put in field 2 in such files. The fields from 5 to 12 above should range from 4 to 11 now.
