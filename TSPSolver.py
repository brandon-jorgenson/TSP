#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
import sys
import random
import math
import copy
from TSPClasses import *
import heapq


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    # Creates a random path that branch and bound uses as the starting bssf
    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        bssf = None
        count = 0

        start_time = time.time()
        for start in range(ncities):
            # Track cities that have been visited
            visited = [False] * ncities
            # Pick an arbitrary start city
            start_city = cities[start]
            visited[start_city._index] = True
            # Construct route
            route = [start_city]
            cost = 0
            for i in range(ncities):
                # Find shortest path to another city
                next_city = None
                next_cost = math.inf
                for j in range(ncities):
                    connection_cost = route[-1].costTo(cities[j])
                    if not visited[j] and connection_cost < next_cost:
                        next_city = cities[j]
                        next_cost = connection_cost
                # Can't go anywhere new
                if next_city == None:
                    cost = math.inf
                    break
                # Add city
                route.append(next_city)
                cost += next_cost
                visited[next_city._index] = True

            # Connect end back to beginning
            cost += route[-1].costTo(start_city)

            count += 1
            if len(route) is ncities and cost is not math.inf:
                solution = TSPSolution(route)
                if not foundTour or  solution.cost < bssf.cost:
                    bssf = solution
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
    	This is the entry point for the algorithm you'll write for your group project.
    	</summary>
    	<returns>results dictionary for GUI that contains three ints: cost of best solution, 
    	time spent to find best solution, total number of solutions found during search, the 
    	best solution found.  You may use the other three field however you like.
    	algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        results = self.greedy()
        coolingRate = .9999
        infiniteDivisor = 0
        iterations = 0
        results['count'] = 0
        results['max'] = 0
        cities = results['soln'].route.copy()
        if len(cities) < 50:
            infiniteDivisor = 1.5
        else:
            infiniteDivisor = 2
        startingTemperature = 100000 * len(cities)
        temperature = startingTemperature

        def chooseNewPath():
            solution = TSPSolution(cities)
            cityIndex = random.randint(0, len(cities) - 1)
            swapTarget = random.randint(0, len(cities) - 1)
            foundSwap = swapTarget
            for i in range(len(cities)):
                swapTarget = (foundSwap + i) % len(cities)
                cities[cityIndex], cities[swapTarget] = cities[swapTarget], cities[cityIndex]
                foundSwap = swapTarget
                if cities[cityIndex].costTo(cities[(cityIndex + 1) % len(cities)]) < np.inf and cities[cityIndex].costTo(cities[(cityIndex - 1 )]) < np.inf:
                    if cities[swapTarget].costTo(cities[(swapTarget + 1) % len(cities)]) < np.inf and cities[swapTarget].costTo(cities[swapTarget - 1]) < np.inf:
                        solution = TSPSolution(cities)
                        break
                cities[cityIndex], cities[swapTarget] = cities[swapTarget], cities[cityIndex]

            def revertNewPath():
                cities[cityIndex], cities[foundSwap] = cities[foundSwap], cities[cityIndex]

            return solution, revertNewPath

        # def chooseNewPath():
        #     solution = TSPSolution(cities)
        #     citiesCopy = cities[:]
        #     cityIndex = random.randint(0, len(citiesCopy) - 1)
        #     swapTarget = cityIndex 
        #     for i in range(len(citiesCopy) - 1):
        #         newSwapTarget = (swapTarget + 1) % len(citiesCopy)
        #         citiesCopy[newSwapTarget], citiesCopy[swapTarget] = citiesCopy[swapTarget], citiesCopy[newSwapTarget]
        #         swapTarget = newSwapTarget
        #         if citiesCopy[newSwapTarget].costTo(citiesCopy[(newSwapTarget + 1) % len(citiesCopy)]) < np.inf and citiesCopy[newSwapTarget].costTo(citiesCopy[(newSwapTarget - 1 )]) < np.inf:
        #             if citiesCopy[swapTarget].costTo(citiesCopy[(swapTarget + 1) % len(citiesCopy)]) < np.inf and citiesCopy[swapTarget].costTo(citiesCopy[swapTarget - 1]) < np.inf:
        #                 solution = TSPSolution(citiesCopy)
        #                 break
        #     return solution

        while time.time() - start_time < time_allowance and temperature > 1:
            newSolution, revertPath = chooseNewPath()
            # May accept a worse path if temperature high
            if newSolution.cost < results['cost'] or np.exp(100*(results['cost'] - newSolution.cost) / temperature) > random.random() or (temperature > startingTemperature / infiniteDivisor and newSolution.cost == np.inf):
                if newSolution.cost != np.inf:
                    results['cost'] = newSolution.cost
                    results['soln'] = TSPSolution(newSolution.route)
                    results['count'] += 1
            else:
                revertPath()
            iterations += 1
            temperature *= coolingRate

        end_time = time.time()
        # Add remaining search states to the number pruned
        print(iterations)
        results['time'] = end_time - start_time

        return results

    # Returns the optimal path between cities, or a possible optimal solution if the function
    # takes longer than the time_allowance to run
    def branchAndBound(self, time_allowance=60.0):
        start_time = time.time()
        results = self.greedy()
        results['count'] = 0
        results['max'] = 0
        results['total'] = 1
        results['pruned'] = 0
        # Generate the first cost matrix and search state and add it to the queue
        cities = self._scenario.getCities()
        originalCostMatrix = self.generateCostMatrix(cities)
        q = []
        path = [cities[0]]
        node = SearchState(0, originalCostMatrix, path)
        node.reduceMatrix(True)
        heapq.heappush(q, (node.cost, node))
        while q and time.time() - start_time < time_allowance:
            parentTuple = heapq.heappop(q)
            parentNode = parentTuple[1]
            # Check if the solution found is better than previous solutions and store it
            if len(parentNode.path) == len(cities):
                if parentNode.cost < results['cost']:
                    results['cost'] = parentNode.cost
                    results['bssf'] = TSPSolution(parentNode.path)
                    results['count'] += 1
                continue
            # Prune search state if the cost is greater than the updated bssf
            if parentNode.cost > results['cost']:
                results['pruned'] += 1
                continue
            # Generate child search states for every possible path
            for i in range(len(cities)):
                if cities[i] not in parentNode.path:
                    results['total'] += 1
                    childPath = parentNode.path.copy()
                    childPath.append(cities[i])
                    childNode = SearchState(parentNode.cost, parentNode.costMatrix, childPath)
                    childNode.reduceMatrix(False)
                    # Add to child state to priority queue if it looks promising, else prune it
                    if childNode.cost < results['cost']:
                        heapq.heappush(q, (childNode.cost / (len(childPath) ** 4), childNode))
                        if len(q) > results['max']:
                            results['max'] = len(q)
                    else:
                        results['pruned'] += 1

        end_time = time.time()
        # Add remaining search states to the number pruned
        results['pruned'] += len(q)
        results['time'] = end_time - start_time
        return results

    # Returns a cost matrix to start the tree
    def generateCostMatrix(self, cities):
        costMatrix = np.zeros([len(cities), len(cities)])
        for x in range(len(cities)):
            origin = cities[x]
            for y in range(len(cities)):
                costMatrix[x, y] = origin.costTo(cities[y])
        return costMatrix


# Defines the nodes of the tree
class SearchState:
    # Every search state takes the cost matrix and cost from its parent and updates them for itself
    def __init__(self, parentCost, costMatrix, path):
        self.cost = parentCost
        self.costMatrix = np.array(costMatrix)
        self.path = path.copy()

    # Comparison operator for the search state so the queue can sort them if the key is the same
    def __eq__(self, other):
        if not isinstance(other, SearchState):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return True

    # Performs a matrix reduction for the search state, updates the cost and matrix
    def reduceMatrix(self, firstState):
        length = len(self.costMatrix)
        # Set rows and columns to infinity if they are child states
        if not firstState:
            startCity = self.path[-2]._index
            endCity = self.path[-1]._index
            self.cost += self.costMatrix[startCity][endCity]
            for col in range(length):
                self.costMatrix[startCity][col] = math.inf
            for row in range(length):
                self.costMatrix[row][endCity] = math.inf
            self.costMatrix[endCity][startCity] = math.inf
        row, col, reduceAmount = 0, 0, 0
        twice = False
        # Reduce rows
        while row < length:
            smallestRowNum = math.inf
            while col < length:
                if twice:
                    self.costMatrix[row][col] -= reduceAmount
                else:
                    if self.costMatrix[row][col] < smallestRowNum:
                        smallestRowNum = self.costMatrix[row][col]
                col += 1
            col = 0
            if smallestRowNum == math.inf:
                smallestRowNum = 0
            if not twice:
                reduceAmount = smallestRowNum
                self.cost += reduceAmount
                twice = True
            else:
                twice = False
                row += 1

        row, col, reduceAmount = 0, 0, 0
        twice = False
        # Reduce columns
        while col < length:
            smallestColNum = math.inf
            while row < length:
                if twice:
                    self.costMatrix[row][col] -= reduceAmount
                else:
                    if self.costMatrix[row][col] < smallestColNum:
                        smallestColNum = self.costMatrix[row][col]
                row += 1
            row = 0
            if smallestColNum == math.inf:
                smallestColNum = 0
            if not twice:
                reduceAmount = smallestColNum
                self.cost += reduceAmount
                twice = True
            else:
                twice = False
                col += 1
        return
