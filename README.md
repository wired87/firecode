# coder

synthax: 
"::" = next jump in local wf
"->" = next pathway step
"-->" = edge connect 
">" bring workflows (wf) = together
"- receive code str ->" = rcs
"receive list prompt" = rlp


legend
similairity search = ss
vectro store = vs

extend and run the following prompt with a clearer workflow definition and detailed tasks to improve the provided Code Module. 


# Relay remote
- receive query string -> create 5 versions of prompt using a local llm (downloadable with pip) > classifier rlp
- qa handling for actions that has been performed. add here qa functionaity in harmony to the terminal input field (render)


# Classifier
rlp -> collect all nodes with type=MODULE ->  perform ss embed_description to 


# researcher 
- - define local vs
  - define static research prompt: gather information based specific isntructions to support the underlying goal
- rlp -> convert to 5 search queries -> gsearch request -> embed and save page content first 5 pages (non ads) in table format within local vs. -> 



# ADD_CODE_GRAPH 
 use ast: identify all present datatypes in each file -> extract and classify all content -> add_node for each with the parent=[parent file-node] and type=datatype(class, def, comment) ´, embed datatype (class/method etc) (only node keys wrapped inside dict(embedding, nid, t, type, code)
- collect all packages specified in r.txt -> add_node(dict(nid, package instance, embed description) 
- scan each datatype for sub modules/classes/functions/motheds/packages used from outer space -> link logical to destination(e.g. method A defines a class inside from external) -> add_edge with rel="uses" - classify directions (seppare between imports or imported)


# Graph Engine
- - create nx.MiltiDiGraph
- walk local dir (exclude .venv) -> add_node all folders type = MODULE, description=embed(llm file request: sum content within  -> extract content each file add_node file_name -> add_edge file_name --> module rel=has_file, ADD_CODE_GRAPH
  





# Collector remote
receive list query
- loop all nodes of the graph embed nid local  perform similarity-serach(ss) ->
  save all in dict(nid:score)
- loop nids with sc > .9:  
  - include a pathfinding algirithmus whcih receivesa nid of a specific datatype e.g. class-> get_node(nid) -> get_neighbors rel="needs" -> collect neighbor nodes global in class attr "self.pathway_nodes"-> repeat process from AAA; output: all nods used by a specific datatype (like class) and allits sub modules  
  - for all params used(method/class header) by specific datatype: collect data from .env file (open...)
  - collect the sorted and runnable code structure incl all variables as str -> return
 
# Executor remote
- rcs -> runnable and executte it by collectig all apckages from code Graph

# editor
- rcs -> llm call gem cli py client: static prompt: perform change on files -> write changes to Graph

# creator 
- rlp -> gem api call: create code base -> ADD_CODE_GRAPH



# ui
**terminal based ui to interact with the engine:**
- incldue state management, issue / submit handling and possibility to query the engine with relay as first contact after query input
- welcome message
- render possible options numbered 
- answer / follow up question handling
  

# Debugger remote
- while loop all files in dir tmp/ray/session_latest: embed content perform classification "err" or "clean" if "err": extract pid from file_name.split(-)-1.split(.)0 -> extract ray actor handle by pid -> get code from remote name (find in graph) -> llm call input: error from file, detailed debug isntructions formulated based on error, . required output: adapted python string -> switch code in graph, keep prev code in "cache":str attr -> delete err file content   

run the extend and inmproved prompt to ensure functionality








# extras:
- use clear oneliner comments before each fuction/method call and at the start of each method to intepret
- use creative prints with emojicons
- include th entire setup to init and run ray 
- include a r.txt (requiremens)
- proviede working ready2use code
- define the entire codebase inside a single file
- each worker must be deployed from 
- define a clear step by step workflow hat executes each possibel cli action wrapped isnide a testing def and if name amin call
- include funcitnoality to load picked datatypes(fucntions & classes) for all defined workflows inisde a ray.remote
- use pyvis for rendering the generated graph file -> save the generated html file in content root
- define each defined workflow inside a ray.remote
- use the followig schema for all add_edge - calls:
  src=,
  trgt=,
  attrs=dict(
      rel=,
      type=,
      trgt_layer=,
      src_layer=,
  ))
