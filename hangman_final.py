import numpy as np

# You are not allowed to import any libraries other than numpy

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF PROHIBITED LIBRARIES WILL RESULT IN PENALTIES

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc

class Node:
    def __init__(self, depth, parent, size, mask = None ):
        self.depth = depth
        self.is_leaf = False
        self.parent = parent
        self.children = {}
        self.actor = None
        self.attr = None
        self.size = size
        self.mask = mask
    
    def train( self, trn_pts_global, trn_pts, node_eval, leaf_eval, min_leaf_size, max_depth, is_pure_enough, get_size ):
        
#       
        if self.size <= min_leaf_size or self.depth >= max_depth or is_pure_enough( trn_pts ):
            self.is_leaf = True
            self.attr = leaf_eval(trn_pts)
        else:
            self.is_leaf = False
            ( self.attr, split_dict ) = node_eval( self.depth, trn_pts, trn_pts_global, self.mask )
#             print(split_dict)

            for ( i, ( outcome, trn_split ) ) in enumerate( split_dict.items() ):

                self.children[ outcome ] = Node( depth = self.depth + 1, parent = self, size = get_size( trn_split ), mask = outcome )

                self.children[ outcome ].train( trn_pts_global, trn_split, node_eval, leaf_eval, min_leaf_size, max_depth, is_pure_enough, get_size )

    
    def get_query(self):
        return self.attr

    def get_child( self, response ):
        
        mask = []
        self_mask = []
        for i in response:
            if i != ' ':
                mask.append(i)
        if self.mask != None:
            for i in self.mask:
                if i != ' ':
                    self_mask.append(i)
        for (j,i) in enumerate(mask):
            if self.mask != None and self_mask[j] != '_':
                mask[j] = self_mask[j]
                
        mask = ' '.join(mask)
             
#         print(self.children.values())
        if mask not in self.children:
            print( "Unseen outcome " +  mask  + " -- using the default_predict routine" )
            exit()
        else:
            return self.children[mask]
                
class Tree:
    def __init__( self, min_leaf_size = 1, max_depth = 15 ):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth

    def train( self, trn_pts_global, node_eval, leaf_eval, is_pure_enough, get_size ):
        
        self.root = Node( depth = 0, parent = None, size = get_size( trn_pts_global ) )
        self.root.train( trn_pts_global, [i for i in range( len(trn_pts_global) )], node_eval, leaf_eval, self.min_leaf_size, self.max_depth, is_pure_enough, get_size )


def get_size(trn_pts):
    return len(trn_pts)

def leaf_eval(trn_pts):
    return trn_pts[0]

def is_pure_enough( trn_pts ):
    return len(trn_pts) <= 1

def node_eval(depth, trn_pts, trn_pts_global, original_mask = None ):
#     if get_size(trn_pts) <= -100:
#         return split_lookahead(trn_pts,trn_pts_global, original_mask)
#     else:
    if depth == 0:
        return split_length( trn_pts, trn_pts_global )
    else :
        return split_no_lookahead( trn_pts, trn_pts_global, original_mask )
    
def split_length( trn_pts, trn_pts_global ):
    
    split_dict = {}
    for index in trn_pts:
        word = trn_pts_global[index]
        mask = [ *( '_' *  len(word)) ]
        mask = ' '.join(mask)
#         print(mask)
        if mask not in split_dict:
            split_dict[mask] = []
        split_dict[mask].append(index)
    return ( "len" , split_dict )

def split_no_lookahead( trn_pts, trn_pts_global, original_mask ):

    best_attr = None
    best_split_dict = {}
#     best_mask = []
    min_entropy = np.inf
    
#     mask_dict = prune_words( trn_pts_global, original_mask )
    
#     for mask in mask_dict:
    for index in trn_pts:
#         index = mask_dict[mask]
        choice = trn_pts_global[index]
        
        ( split_dict, entropy ) = try_attr( trn_pts, choice, trn_pts_global, original_mask )
        
        if entropy < min_entropy:
            min_entropy = entropy
            best_split_dict = split_dict
            best_attr = index
#             best_mask = curr_mask
            
    return ( best_attr, best_split_dict )

def try_attr( trn_pts, choice, trn_pts_global, original_mask ):
    
    count_dict = {}
    split_dict = {}
    for index in trn_pts:
        word = trn_pts_global[index]
        curr_mask = []
        for i in original_mask:
            if i != ' ':
                curr_mask.append(i)
        
        for j in range( min( len(choice), len(word) ) ):
            if curr_mask[j] == '_' and word[j] == choice[j]:
                curr_mask[j] = choice[j]
        
        curr_mask = ' '.join(curr_mask)
        if curr_mask not in split_dict:
            count_dict[curr_mask] = 0
            split_dict[curr_mask] = []
            
        count_dict[curr_mask] += 1
        split_dict[curr_mask].append(index)
        
    entropy = get_entropy( np.array( list( count_dict.values() ) ) )
    return ( split_dict, entropy )

def prune_words( trn_pts_global, original_mask ):
    mask_dict = {}
    for ( i, word ) in enumerate( trn_pts_global ):
        mask = []
        for j in original_mask:
            if j != ' ':
                mask.append(j)
        for j in range( min( len(mask), len(word) ) ):
            if mask[j] == '_':
                mask[j] = word[j]
        mask = ' '.join(mask)
        if mask not in mask_dict:
            mask_dict[mask] = i
    return mask_dict

def get_entropy( counts ):    
    assert np.min( counts ) > 0, "Elements with zero or negative counts detected"
    
    num_elements = counts.sum()
    
    if num_elements <= 1:
        print( f"warning: { num_elements } elements in total." )
        return 0
    
    proportions = counts / num_elements
    
    return np.sum( proportions * np.log2( counts ) )  



################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your decision tree model using the word list provided
	# Return the trained model as is -- do not compress it using pickle etc
	# Model packing or compression will cause evaluation failure

	model = Tree()
	model.train( words, node_eval, leaf_eval, is_pure_enough, get_size )
	return model					# Return the trained model
