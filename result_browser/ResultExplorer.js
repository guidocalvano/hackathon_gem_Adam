

define([], function() {

    function objectToHierarchy(obj) {
        if (!(obj instanceof Object)) return []

        return Object.keys(obj).map(key => {
            let subHierarchy = objectToHierarchy(obj[key])

            let node = {
                name: key,
                children: subHierarchy
            }

            if (subHierarchy.length === 0) node.targetData = obj[key]


            return node
        })
    }

    function updateTable(headerNode, rowsNode, dataUrl) {
        let numberFormat = d3.format('.3n');

        d3.csv(dataUrl).then(data => {
            var t = d3.transition()
                .duration(750)
                .ease(d3.easeLinear);

            window.data = data
            let maxData = data.reduce((max, row) => Math.max(max, Object.values(row).reduce((rowMax, v) => Math.max(rowMax, isNaN(parseFloat(v)) ? -Infinity : parseFloat(v)), -Infinity)), -Infinity)
            let numberColorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxData])
            let colorScale = v => isNaN(parseFloat(v)) ? 'rgb(240, 240, 255)' : numberColorScale(parseFloat(v))


            console.log(JSON.stringify(data.columns))
            let headerMatches = headerNode
                .selectAll('.header')
                .data(data.columns)

            let newheaders = headerMatches.enter()
                .append('g')
                .attr('class', 'header')
                .attr('transform', (d, i) => 'translate(' + (i * 90) + ', 0)')

            newheaders.style('opacity', 0)
                .transition(t)
                .style('opacity', 1)

            newheaders
                .append('rect')
                .attr('width', 90)
                .attr('height', 40)
                .attr('fill', 'rgb(20, 20, 0)')

            newheaders
                .append('text')
                .attr('font-family', 'helvetica')
                .attr('fill', 'rgb(240, 240, 255)')
                .attr('y', 20)
                .text(d => d)

            headerMatches.select('text').text(d => d)

            headerMatches.exit().transition(t).style('opacity', 0).remove()

            let rowMatch = rowsNode.selectAll('.row').data(data.slice())
            rowEnter = rowMatch.enter()
                .append('g')
                .attr('class', 'row')
                .attr('transform', (d, i) => 'translate(0, ' + ((i + 1) * 40) + ')');

            rowMatch.exit().transition(t).style('opacity', 0).remove()

            function createNewDataCell(mountNode) {
                let newDataCell = mountNode.append('g')
                    .attr('class', 'cell')
                    .attr('transform', (d, i) => 'translate(' + (i * 90) + ', 0)')

                newDataCell.append('rect')
                    .attr('width', 90)
                    .attr('height', 40)
                    .style('opacity', 0)
                    .transition(t)
                    .style('opacity', 1);
                // .attr('fill', 'rgb(240, 240, 255)');
                newDataCell.append('text')
                    .attr('y', '15')
                    .attr('font-family', 'helvetica')
                    .style('opacity', 0)
                    .transition(t)
                    .style('opacity', 1);

                newDataCell.call(updateCell)
            }

            function updateCell(mountNode, d, i) {
                mountNode.select('rect').transition(t).attr('fill', d => colorScale(d))
                mountNode.select('text')
                    .text((d, i) => i > 0 ? numberFormat(d) : d);
            }

            let cellMatch = rowEnter.selectAll('.cell').data(d => data.columns.map(c => d[c]))


            cellMatch.enter().call(createNewDataCell)

            let rowUpdateCellMatch = rowMatch.selectAll('.cell').data(d => data.columns.map(c => d[c]))

            rowUpdateCellMatch.enter().call(createNewDataCell)

            rowUpdateCellMatch.call(updateCell)

            rowUpdateCellMatch.exit().transition(t).style('opacity', 0).remove()
        })

    }


    //return an object to define the "my/shirt" module.
    return (mountNode, data_url) => {
        let tableGroup = d3.select(mountNode)
            .append('g')
            .attr('class', 'resultsTable')
            .attr('transform', 'translate(500, 0)');

        let headerGroup = tableGroup.append('g').attr('class', 'header');
        let rowsGroup = tableGroup.append('g').attr('class', 'rows')

        d3.json(data_url).then(function (data) {
            let treeGroup = d3.select(mountNode).append('g').attr('transform', 'translate(50, 200)')
            let nodeGroup = treeGroup
                .append('g')
                .attr("fill", "none")
                .attr("stroke", "#555")
                .attr("stroke-opacity", 0.4)
                .attr("stroke-width", 1.5);

            let linkGroup = treeGroup
                .append('g')
                .attr("fill", "none")
                .attr("stroke", "#555")
                .attr("stroke-opacity", 0.4)
                .attr("stroke-width", 1.5);

            let tree = d3.tree().nodeSize([15, 70]);
            let diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

            window.data = data;
            let hierarchy = {name: "root", children: objectToHierarchy(data)}
            window.hierarchy = hierarchy
            let root = d3.hierarchy(hierarchy)

            tree(root);

            window.root = root;
            const nodeData = root.descendants().reverse();

            const nodeElements = nodeGroup.selectAll("g")
                .data(nodeData, d => d.id);

            let nodeContainerEnter = nodeElements.enter()
                .append('g')
                .attr('class', 'nodeContainer')
                .attr('transform', d => 'translate(' + d.y + ', ' + d.x + ')');

            nodeContainerEnter
                .append('circle')
                .attr('r', 3.4)
            nodeContainerEnter
                .append('text')
                .attr('font-family', 'helvetica')
                .text(d => d.data.name)
                .attr('stroke', 'black')
                .attr('fill', 'black')
                .on('click', d => d.data.targetData && updateTable(headerGroup, rowsGroup, d.data.targetData))

            let linkData = root.links();


            let linkElements = linkGroup.selectAll("path")
                .data(linkData, d => d.target.id);


            let linkEnter = linkElements.enter().append("path")
                .attr("d", d => {

                    return diagonal(d);
                });

        });

    }


});
