# Framework para abstracción de entornos heterogéneos CPU/GPU

## Optimización de renderización por cálculo de occlusion culling en paralelo



​		En este proyecto se desarrollará un framework que proporcione recursos de computación a las aplicaciones de manera que les permita aislarse de los detalles de bajo nivel a la vez que les proporcione de herramientas para poder adaptar la asignación de los recursos a sus necesidades reales. En este caso nos centraremos en intentar optimizar el la optimización del proceso de renderizado de escenas obviando aquellos objetos que estén ocultos por otros de forma que no se desperdicie tiempo en cálculos que luego no serán realmente útiles.

​		Este trabajo viene de la motivación provocada por la falta de contenido abierto que existe en el ámbito de los gráficos, si bien es cierto que existen librerías que ayudan a la realización de dicha tarea, todas suelen ser de código cerrado o soluciones que hayan implementado directamente los desarrolladores de los distintos motores gŕaficos que se utilizan en la actualidad ya sea Unreal Engine o Unity. Los cuales aún siendo gratuitos de usar, no podemos saber qué algoritmos/tecnologías los componen ni si pueden ser mejoradas o adaptadas para las próximas tecnologías. 

![main](https://raw.githubusercontent.com/jcpulido97/TFG/master/doc/img/screenshot.png?token=AFM4SFGQLZNL2OPI63JE7AC5CCQRA)

### Estructura

- **doc/** se encuentra toda la documentación del proyecto tanto diagramas como imágenes
- **include/** todos los ficheros de declaración del proyecto
- **src/** todos los ficheros de implementación del proyecto
- **tests/** todos los tests de las diferentes clases del proyecto
- **libs/** las librerías usadas en este proyecto. En este caso solo [MathGeoLib](https://github.com/juj/MathGeoLib)

​	

### Mejoras
Se han ejecutado distintas pruebas en el siguiente equipo:

| Categoría             | Dispositivo            |
| --------------------- | ---------------------- |
| **CPU**               | i7-7700HQ              |
| **Cantidad de RAM**   | 16 GB                  |
| **GPU**               | NVIDIA GTX 1050        |
| **Sistema Operativo** | Ubuntu 18.04           |
| **GPU Drivers**       | NVIDIA drivers 390.116 |

![Mejoras](https://raw.github.com/jcpulido97/TFG/master/doc/img/prune_benchmark.svg?sanitize=true)

Se obtienen bastantes mejoras usando el software para evitar renderizar objetos que están ocluidos

![teoric_limit](https://raw.github.com/jcpulido97/TFG/master/doc/img/teoric_limit.svg?sanitize=true)

Mis recomendación es que no se dude en usarlo cuando tengamos escenas de 500 objetos o menos y que cuando la cantidad de objetos sea mayor a 500 hagamos una prueba de si de verdad merece la pena para nuestro caso específico. Destacar que todas las pruebas han sido ejecutadas en el siguiente entorno

