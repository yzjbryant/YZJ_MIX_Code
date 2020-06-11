states_needed = set(["mt","wa","or","id","nv","ut","ca","az"])
#你传入一个数组，它被转换为集合

arr = [1,2,2,3,3,3]
set(arr)
set([1,2,3])

stations = {}
stations["kone"] = set(["id","nv","ut"])
stations["ktwo"] = set(["wa","id","mt"])
stations["kthree"] = set(["or","nv","ca"])
stations["kfour"] = set(["nv","ut"])
stations["kfive"] = set(["ca","az"])

final_stations = set()

best_station  = None
states_covered = set()
for station, states_for_station in stations.items():
    covered = states_needed & states_for_station  #你从来没见过的语法！它计算交集
    if len(covered) > len(states_covered):
        best_station = station
        states_covered = covered
        final_stations.add(best_station)
        states_needed -= states_covered

        while states_needed:
            best_station = None
            states_covered = set()
            for station, states in stations.items():
                covered = states_needed & states
                if len(covered) > len(states_covered):
                    best_station = station
                    states_covered = covered
                    states_needed -= states_covered
                    final_stations.add(best_station)
                    print(final_stations)