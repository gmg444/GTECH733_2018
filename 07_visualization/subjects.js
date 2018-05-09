var subjectDict = {
	'SA':	'Arts and culture',
	'SB':	'Education',
	'SC':	'Environment',
	'SD':	'Philanthropy',
	'SE':	'Health',
	'SF':	'Science',
	'SG':	'Social sciences',
	'SH':	'Information', //  and Communications
	'SJ':	'Public safety',
	'SK':	'Public affairs',
	'SM':	'Agriculture', //, fishing and forestry
	'SN':	'Community', //  and economic development
	'SP':	'Religion',
	'SQ':	'Sports', //  and recreation
	'SR':	'Human rights',
	'SS':	'Human services',
	'ST':	'International', //  relations
	'SZ':	'Unknown' //  or not classified
};

// SubjectToSubjectDelta.txt
// SubjectToSubject.txt
// SubjectToSubjectLevel2.txt
// SubjectToSubjectDeltaNormalized.tx

d3.tsv("subject_to_subject_mac.txt", function(subjectToSubject){
	var i;
	var nodes = []
	var edges = []
	for(i=0; i<subjectToSubject.length; i++){
		node1name = subjectToSubject[i].id1.split(" ").join("_");
		node2name = subjectToSubject[i].id2.split(" ").join("_");
		
		if (nodes.indexOf(node1name) < 0){
			nodes.push(node1name);
		}
		if (nodes.indexOf(node2name) < 0){
			nodes.push(node2name);
		}
		var e = {};
		e["node1"] = node1name;
		e["value"] = subjectToSubject[i].count;
		e["node2"] = node2name;
		edges.push(e);
	}

	m = createMatrix(nodes, edges);

	visualize(m, nodes);

});

function initMatrix(size){
	var i, j;
	var matrix = [];
	var row;

	matrix.length = size;
	for (i=0; i<size; i++){
		row = matrix[i] = [];
		row.length = size;
		for (j=0; j<size; j++){
			row[j] = 0;
		}
	}
	return matrix;
}; 

function createMatrix(nodes, edges){
	var matrix = initMatrix(nodes.length);
	var i = 0;
	var index = {};
	for (i=0; i<nodes.length; i++){
		index[nodes[i]] = i;
	}

	var row;
	var irow, icol;

	for(i=0; i<edges.length; i++){
		row = edges[i];
		icol = index[row.node1];
		irow = index[row.node2];
		if (irow != icol){
			matrix[irow][icol] = parseInt(row.value);
		}
	}
	return matrix;
};

function visualize(m, nodes){

	// Create svg container
	var width = 720, height = 720;
	var svg = d3.select("body").append("svg")
		.attr("width", width)
		.attr("height", height)
		.append("g")
			.attr("id", "circle")
			.attr("transform", "translate(" + width/2 + "," + height/2 + ")");

	// Prepare layout
	var layout = d3.layout.chord()
		.padding(.04)
		.sortSubgroups(d3.descending)
		.sortChords(d3.ascending)
		.matrix(m);

	var chord;
	// Set up event for chord groups
	var group = svg.selectAll(".group").data(layout.groups).enter()
		.append("g").attr("class", "group").on("mouseover",
			function mouseover(d, i){
				chord.classed("fade", function(d){
					return d.source.index != i && d.target.index != i;
				});
			});

	// Set colors for perimeter of circle
	var startHue = 180, hueStep = 360 / nodes.length;
	var radius = Math.min(width, height) / 2 - 10,
		innerRadius = radius - 24;

	function nodeColor(i, l){
		return d3.hsl(startHue - i*hueStep, 0.7, l || 0.5).rgb().toString();
	}

	// Draw the arcs of the circle
	var groupPath = group.append("path").attr("id", function(d, i){
		return "group" + i;
	}).attr("d", d3.svg.arc().innerRadius(innerRadius).outerRadius(radius))
		.style("fill", function(d, i){
			return nodeColor(i);
		})
		.style("stroke", function(d, i){
			return nodeColor(i, 0.33);
		});


	// Add labels
	var labelXOffset = 3, labelYOffset = 15;

	var groupText = group.append("text")
		.attr('x', labelXOffset)
		.attr("dy", labelYOffset);

	groupText.append("textPath").attr('xlink:href', function (d, i){
		return "#group" + i;
	}).text(function(d, i){
		if (d.endAngle - d.startAngle < 0.2)
			return nodes[i].substring(0, 2);
		else if (d.endAngle - d.startAngle < 0.4)
			return nodes[i].substring(0, 10);
		else{
			return nodes[i].substring(0, 20).split("_").join(" ")
		}
	}); 

	// Create the chords letting D3 do the work
	chord = svg.selectAll(".chord").data(layout.chords).enter()
		.append("path").attr("class", "chord")
		.attr("d", d3.svg.chord().radius(innerRadius))
		.style("fill", function(d){
			return nodeColor(d.source.index);
		})
		.style("stroke", function(d){
			return nodeColor(d.source.index, 0.25);
		});

	var formatValue = d3.format(',f');
	svg.insert('circle', ':first-child').attr('r', radius);

	group.append('title').text(
		function(d, i){
			return nodes[i].split("_").join(" ") + ": " + formatValue(d.value);
		}
	);

	chord.append('title').text(
		function(d){
			return nodes[d.source.index].split("_").join(" ") + ' : ' + nodes[d.target.index].split("_").join(" ");
		}
	);
};
