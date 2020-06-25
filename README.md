# link-prediction
## Data Description
<p>The training network is a fragment of an academic co-authorship graph. The nodes in the network—authors— have been given randomly assigned IDs, and an undirected edge between node A and B represents that authors A and B have published a paper together as co-authors. The training network is a network of a time period (2010-2017), focusing on individuals in a specific academic sub-community.</p>
  
### train.txt
<p>txt file containing training graph data  in a (tab delimited) edge list format, where each row represents a node and its neighbors. For example:</p>
1 2 <br />
2 1  3  4 <br />
3 2  5 <br />
4 2  5 <br />
5 3  4 <br />
  
### nodes.json
<p>JSON file including several features of the nodes (authors).</p>
<ul>
<li>their id in the graph </li>
<li>the number of years since their first and last publication to 2017 (e.g. first:3 means author published first paper at 2014) </li>
<li>their number of publications in total, num_papers </li>
<li>presence of specific keywords in the titles and abstracts of their publications (denoted keyword_X where X ∈ {0, 1,..., 53}, each being a binary value and only listed if its value is 1) </li>
<li>publication at specific venues (denoted venue_X where X ∈ {0, 1,..., 303}, each being a binary value and only listed if its value is 1) </li>
</ul>
  
### dev.csv & dev-labels.csv
<p>The development set is a list of 4,866 edges, contain 2,433 real edges in the year after the time period of the training set (2018) , and also 2,433 fake edges (pairs of nodes that are not connected).</p>

## Model
### GAE
Require: <br />
Tenserflow (1.0 or latter)
