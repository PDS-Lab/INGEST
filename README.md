# INGEST


## Build
```
cmake -B build -G "Ninja Multi-Config" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --config Release
```

## Run
```
./build/Release/mem-server.exe -p xxx -l 2 -m 3 1:80g   # mem server

./build/Release/mem-init.exe /path/to/init_graph 80g addr:port:1 ...

./build/Release/bench-insert-search-n.exe start-nvecs end-nvecs step-nvecs base-path query-path gt-path global-barrier-addr graph-cfg lock-cfgs client-rank nclients
```