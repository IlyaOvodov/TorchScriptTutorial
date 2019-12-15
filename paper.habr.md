Около года назад разработчики PyTorch представили сообществу **TorchScript** - инструмент, который позволяет с помощью пары строк кода и нескольких щелчков мыши сделать из пайплайна на питоне отчуждаемое решение, которое можно встроить в систему на  C++. Ниже я делюсь опытом его использования и постараюсь описать встречающиеся на этом пути подводные камни. Особенное внимание уделю реализации проекта на Windows, поскольку, хотя исследования в ML обычно делаются на Ubuntu, конечное решение часто (внезапно!) требуется под "окошками". 

Примеры кода для экспорта модели и проекта на C++, использующего модель, можно найти в [репозиториии на GitHub](https://github.com/IlyaOvodov/TorchScriptTutorial).

[![](https://habrastorage.org/webt/3k/u1/ub/3ku1ubmzigl3j016ezncczdonqm.jpeg)](https://habr.com/ru/company/ods/blog/480328/#continue)

<cut/>
<anchor>continue</anchor>

Разработчики PyTorch не обманули. Новый инструмент действительно позволяет превратить исследовательский проект на PyTorch в код, встраиваемый в систему на С++, за пару рабочих дней, а при некотором навыке и быстрее.

TorchScript появился в PyTorch версии 1.0  и продолжает развиваться и меняться. Если первая версия годичной давности была полна багов и являлась скорее экспериментальной, то актуальная на данный момент версия 1.3 как минимум по второму пункту заметно отличается: экспериментальной ее уже не назовешь, она вполне пригодна для практического использования. Я буду ориентироваться на нее.

В основе TorchScript лежит собственный автономный (не требующий наличия Python) компилятор питон-подобного языка, а также средства для конвертации в него программы, написанной на Python и PyTorch, методы сохранения и загрузки получившихся модулей и библиотека для их использования в C++. Для работы придется добавить в проект несколько DLL общим весом около 70MB (для Windows) для работы на CPU и 300MB для GPU версии. TorchScript поддерживает большинство функций PyTorch и основные возможности языка python. А вот о сторонних библиотеках, таких как OpenCV или NumPy, придется забыть. К счастью, у многих функций из NumPy есть аналог в PyTorch.

## Конвертируем пайплайн на PyTorch модель на TorchScript

TorchScript предлагает два способа преобразования кода на Python в его внутренний формат: tracing и scripting (трассировка и скриптование). Зачем два? Нет, понятно, конечно, что два лучше чем один...

![](https://habrastorage.org/webt/lh/xp/ww/lhxpwwynynljq2_sxj35jhpp9yc.jpeg)

Но в случае с этими методами получается как в известном афоризме про левый и правый уклон: оба хуже. Что ж, мир несовершенен. Просто в конкретной ситуации надо выбирать тот, который больше подходит.

Метод трассировки очень прост. Берется некий образец данных (обычно инициализированный случайными числами), отправляется в интересующую нас функцию или метод класса, и PyTorch строит и запоминает граф вычислений примерно так же, как делает это обычно при обучении нейросети. Вуаля - скрипт готов:

```python
import torch
import torchvision
model = torchvision.models.resnet34(pretrained = True)
model.eval()
sample = torch.rand(1, 3, 224, 224)
scripted_model = torch.jit.trace(model, sample)
```

В примере выше получается объект класса ScriptModule. Его можно сохранить

```python
scripted_model.save('my_script.pth')
```

и загрузить потом в [программу на C++](https://github.com/IlyaOvodov/TorchScriptTutorial/tree/master/cpp_proj) (об этом [ниже](#cpp)) или в код на Python вместо исходного объекта:

<spoiler title="Пример кода на Python, использующего сохраненную модель">
```python
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize
transforms = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img = cv2.resize(cv2.imread('pics/cat.jpg'), (224,224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = transforms(img).unsqueeze(0) # add batch dimension

scripted_model = torch.jit.load('my_script.pth')
y = scripted_model(x)

print(y[0].argmax(), y[0][y[0].argmax()])
```

    tensor(282) tensor(12.8130, grad_fn=<SelectBackward>)
</spoiler>

Получающийся объект `ScriptModule` может выступать везде, где обычно используется `nn.Module`.

Описанным способом можно трассировать экземпляры класса `nn.Module` и функции (в последнем случае получается экземпляр класса `torch._C.Function`).

Этот метод (tracing) имеет важное преимущество: так можно конвертировать почти любой питоновский код, не использующий внешних библиотек. Но есть и не менее важный недостаток: при любых ветвлениях будет запомнена только та ветка, которая исполнялась на тестовых данных:

```python
def my_abs(x):
    if x.max() >= 0:
        return x
    else:
        return -x
my_abs_traced = torch.jit.trace(my_abs, torch.tensor(0))
print(my_abs_traced(torch.tensor(1)), my_abs_traced(torch.tensor(-1)))
```
    c:\miniconda3\lib\site-packages\ipykernel_launcher.py:2: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!

    tensor(1) tensor(-1)
    
Упс! Кажется, это не то, что мы хотели бы, правда? Хорошо, что по этому поводу хотя бы выдаётся предупреждающее сообщение (TracerWarning). Относиться к таким сообщениям стоит внимательно.

Тут нам на помощь приходит второй метод - scripting:

```python
my_abs_script = torch.jit.script(my_abs)
print(my_abs_script(torch.tensor(1)), my_abs_script(torch.tensor(-1)))
```
    tensor(1) tensor(1)
    
Ура, ожидаемый результат получен! Scripting рекурсивно анализирует код на Python и преобразует в код на собственном языке. На выходе получаем тоже класс `ScriptModule` (для модулей) или `torch._C.Function`(для функций) . Казалось бы, вот оно, счастье! Но возникает другая проблема: внутренний язык TorchScript строго типизированный, в отличие от Python. Тип каждой переменной определяется первым присваиванием, тип аргументов функции по умолчанию - `Tensor`. Поэтому, например, привычный шаблон 

```python
def my_func(x):
    y = None
    if x.max() > 0:
        y = x
    return y
my_func = torch.jit.script(my_func)
```

оттрассировать не удастся.

<spoiler title="Ошибка трассировки выглядит так">

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-9-25414183a687> in <module>()
    ----> 1 my_func = torch.jit.script(my_func)

    d:\programming\3rd_party\pytorch\pytorch_ovod_1.3.0a0_de394b6\torch\jit\__init__.py in script(obj, optimize, _frames_up, _rcb)
       1224         if _rcb is None:
       1225             _rcb = _gen_rcb(obj, _frames_up)
    -> 1226         fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
       1227         # Forward docstrings
       1228         fn.__doc__ = obj.__doc__

    RuntimeError: 
    Variable 'y' previously has type None but is now being assigned to a value of type Tensor
    :
    at <ipython-input-8-75677614fca6>:4:8
    def my_func(x):
        y = None
        if x.max() > 0:
            y = x
            ~ <--- HERE
        return y

</spoiler>
    
Примечательно, что, хотя ошибка возникает при вызове `torch.jit.script`, указывается и вызвавшее ее место в скриптуемом коде.

Даже точки после констант начинают играть роль:

```python
def my_func(x):
    if x.max() > 0:
        y = 1.25
    else:
        y = 0
    return y
my_func = torch.jit.script(my_func)
```

<spoiler title="выдаст ошибку">

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-10-0a5f18586763> in <module>()
          5         y = 0
          6     return y
    ----> 7 my_func = torch.jit.script(my_func)
    
    d:\programming\3rd_party\pytorch\pytorch_ovod_1.3.0a0_de394b6\torch\jit\__init__.py in script(obj, optimize, _frames_up, _rcb)
       1224         if _rcb is None:
       1225             _rcb = _gen_rcb(obj, _frames_up)
    -> 1226         fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
       1227         # Forward docstrings
       1228         fn.__doc__ = obj.__doc__
    
    d:\programming\3rd_party\pytorch\pytorch_ovod_1.3.0a0_de394b6\torch\jit\__init__.py in _rcb(name)
       1240         # closure rcb fails
       1241         result = closure_rcb(name)
    -> 1242         if result:
       1243             return result
       1244         return stack_rcb(name)
    
    RuntimeError: bool value of Tensor with more than one value is ambiguous

</spoiler>

Потому что надо писать не `0`, а `0.`, чтобы тип в обеих ветках был одинаковым! Избаловались, понимаешь, со своим питоном!

Это только начало списка тех изменений, которые требуется внести в код на python, чтобы его можно было успешно превратить в модуль TorchScript. Более подробно самые типичные случаи перечислю [чуть позже](#tips). В принципе, никакой rocket science тут нет и свой код вполне можно поправить соответствующим образом. А вот исправлять сторонние модули, включая стандартные из `torchvision`, чаще всего править не хочется, а "как есть" для скриптования они обычно не пригодны.

К счастью, обе технологии можно совмещать: то, что скриптуется - скриптовать, а что не скриптуется - трассировать:

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained = True)
        # без следующих двух строк попытка сделать torch.jit.script(my_module)
        # ниже выдаст ошибку где-то в недрах resnet34.
        # Поэтому заблаговременно сами заменим self.resnet на ScriptModule.
        self.resnet.eval() # NB: это надо сделать до трассировки! После - не сработает!
        self.resnet = torch.jit.trace(self.resnet, torch.rand((1,3,224,224),
                                      dtype=torch.float))
    def forward(self, x):
        if x.shape[2] < 224 or x.shape[3] < 224:
            return torch.tensor(0)
        else:
            return self.resnet(x)
my_module = MyModule()
my_module = torch.jit.script(my_module)
```

В примере выше трассировка используется, чтобы включить модуль, не поддающийся скриптованию, в модуль, где не достаточно трассировки и необходимо скриптование. Бывает и обратная ситуация. Например, если нам надо выгрузить модель в ONNX, при этом используется трассировка. Но трассируемая модель может включать функции на TorchScript, поэтому логику, требующую ветвлений и циклов, можно реализовать там! Пример приведен в [официальной документации по torch.onnx](https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting).

Более подробно возможности, предоставляемые PyTorch для создания модулей на TorchScript описаны в [официальной документации](https://pytorch.org/docs/stable/jit.html) и [руководстве](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) по `torch.jit`. В частности, я не упомянул об удобном способе использования `torch.jit.trace` и `torch.jit.script` в виде декораторов, об особенностях отладки скриптованного кода. Это и многое другое есть в документации.

## <anchor>cpp</anchor>Включаем модель в проект на C++

К сожалению, [официальная документация](https://pytorch.org/tutorials/advanced/cpp_export.html) ограничивается примерами вида "сложить 2 тензора, сгенерированных с помощью `torch.ones`". Я подготовил пример [более приближенного к реальности проекта](https://github.com/IlyaOvodov/TorchScriptTutorial/tree/master/cpp_proj), отправляющего в нейросеть картинку из OpenCV и получающего обратно результаты в виде тензора откликов, кортежа переменных, картинки с результатами сегментации.

Для работы примера потребуются сохраненные скрипты классификации c помощью ResNet34 и сегментации с помощью DeepLabV3. Для подготовки этих скриптов надо запустить [этот jupyter блокнот](https://github.com/IlyaOvodov/TorchScriptTutorial/blob/master/prepare_scripts.ipynb).

Нам потребуется библиотека `torchlib`. Получить ее можно несколькими путями:
1. Если у вас уже стоит PyTorch, поставленный с помощью `pip install`, то ее можно найти в каталоге Python: `<Miniconda3>\Lib\site-packages\torch`;
2. Если у вас PyTorch собран из исходников, то она там: `<My Pytorch repo>\build\lib.win-amd64-3.6\torch`;
3. Наконец, можно скачать с [pytorch.org](https://pytorch.org) отдельно библиотеку, выбрав Language = C++, и распаковать архив.

Код на C++ достаточно прост. Надо:
1. Включить заголовочный файл
```C++
#include <torch/script.h>
```
2. Загрузить модель
```C++
torch::jit::script::Module module = torch::jit::load("../resnet34_infer.pth");
```
3. Подготовить данные
```C++
torch::Tensor tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
```
4. Вызвать фунцию `forward` и получить результат
```C++
auto output = module.forward( { tensor } )
```
5. Получить данные из результата. Как это сделать, зависит от того, что возвращает нейросеть. Кстати, принимать она в общем случае может тоже не только одну картинку, поэтому лучше посмотреть [на исходный код примера](https://github.com/IlyaOvodov/TorchScriptTutorial/blob/master/cpp_proj/cpp_proj.cpp) целиком, там присутствуют разные варианты. Например, для получения данных из одномерного тензора типа float:
```C++
float* data = static_cast<float*>(output.toTensor().data_ptr());
```
6. Есть еще одна тонкость. Не забыть вставить в код аналог `with torch.no_grad()`, чтобы не тратить ресурсы на вычисление и хранение не нужных нам градиентов. К сожалению, эту команду нельзя включить в скрипт, поэтому приходится добавлять в код на С++:
```C++
torch::NoGradGuard no_grad;
```

Как собрать проект с помощью CMake, описано в [официальном руководстве](https://pytorch.org/tutorials/advanced/cpp_export.html). А вот тема проекта на Visual Studio там не раскрыта, поэтому опишу это подробнее. Придется вручную подкрутить настройки проекта:
1. Я тестировал на Visual Studio 2017. Про другие версии сказать не могу.
2. Должна быть установлена v14.11 тулсета v141 (галочка `"VC++ 2017 version 15.4 v14.11 toolset"` в инсталляторе VS).
3. Платформа должна быть `x64`.
4. В `General → Platform Toolset` выбрать `v141(Visual Studio 2017)`
5. В `C/C++ → General → Additional Include Directories` добавить `<libtorch dir>\include`
6. В `Linker → General → Additional Library Directories` добавить `<libtorch dir>\lib`
7. В `Linker → Input → Additional Dependencies` добавить `torch.lib; c10.lib`. В интернетах пишут, что еще может потребоваться `caffe2.lib`, а для GPU и еще что-нибудь из `<libtorch dir>\lib`, но в текущей версии мне хватало добавления этих двух библиотек. Возможно, это устаревшая информация.
8. Пишут также, что надо ставить `C/C++ → Language → Conformance Mode` = `No`, но я не увидел разницы.

Также в проекте НЕ должна быть объявлена переменная `__cplusplus`. Попытка добавить [опцию компилятора `/Zc:__cplusplus`](https://docs.microsoft.com/ru-ru/cpp/build/reference/zc-cplusplus?view=vs-2017) приведет к ошибкам при компиляции в файле `ivalue.h`.

В [прилагаемом проекте](https://github.com/IlyaOvodov/TorchScriptTutorial/tree/master/cpp_proj) настройки путей (не только к TorchLib, но и к OpenCV и CUDA) вынесены в [props файл](https://github.com/IlyaOvodov/TorchScriptTutorial/blob/master/cpp_proj/cpp_proj.props), перед сборкой надо прописать их там в соответствии с вашей локальной конфигурацией. Вот, собственно, и все.

## <anchor>tips</anchor>Что ещё следует иметь в виду

Если описанный процесс показался вам слишком простым, интуиция вас не обманула. Есть целый ряд нюансов, которые надо учитывать, чтобы преобразовать модель на PyTorch, написанную на Python, в TorchScript. Перечислю ниже те, с которыми приходилось сталкиваться. Некоторые я уже упоминал, но повторюсь, чтобы собрать все в одном месте.

![](https://habrastorage.org/webt/iv/xy/q-/ivxyq-lqqw8s1aqd_cy4t4uwj5i.jpeg)

* Типом переменных, передаваемых в функцию, по умолчанию считается Tensor. Если в каких-то (весьма частых) случаях это окажется неприемлемым, придется объявить типы вручную, используя MyPy-style type annotations, примерно так:

```python
def calc_letter_statistics(self, cls_preds: List[Tensor], cls_thresh: float)->Tuple[int, Tuple[Tensor, Tensor, Tensor]]
```
или так: 
```python
def calc_letter_statistics(self, cls_preds, cls_thresh):
    # type: (List[Tensor], float)->Tuple[int, Tuple[Tensor, Tensor, Tensor]]
```

*  Переменные строго типизированы и тип, если не указан явно, определяется первым присваиванием. Привычные конструкции вида `x=[]; for ...:  x.append(y)` придется отредактировать, т.к. в момент присваивания `[]` компилятор не может понять, какой тип будет в списке. Поэтому придется указать тип явно, например:

```python
from typing import List
x: List[float] = []
```
или (другое "например")
```python
from torch import Tensor
from typing import Dict, Tuple, List
x: Dict[int: Tuple[float, List[Tensor], List[List[int]]]] = {}
```

* В примере выше надо импортировать именно имена, поскольку эти имена зашиты в код TorchScript. Альтернативный, казалось бы, законный, подход

```python
import torch
import typing
x: typing.List[torch.Tensor] = []
```
приведет при скриптовании к ошибке *Unknown type constructor typing.List*

* Еще одна привычная конструкция, с которой придется расстаться:

```python
x = None
if smth:
    x = torch.tensor([1,2,3])
```
Тут есть два варианта. Или оба раза присваивать Tensor (то, что он разной размерности, не страшно):
```python
x = torch.tensor(0)
if smth:
    x = torch.tensor([1,2,3])
```
и не забыть поискать, что сломается после такой замены. Или попытаться честно написать:
```python
x: Optional[Tensor] = None
if smth:
    x = torch.tensor([1,2,3])
```
но тогда при дальнейшем использовании `x` там, где ожидается тензор, мы, скорее всего, получим ошибку: *Expected a value of type 'Tensor' for argument 'x' but instead found type 'Optional[Tensor]'.*

* Не забываем при первом присваивании писать, например, `x=0.` вместо привычного `x=0` и т.п., если переменная `x` должна иметь тип `float`.

* Если где-то использовалась старомодная инициализация тензора через `x = torch.Tensor(...)`, с ней придется расстаться и заменить на более молодежный вариант с маленькой буквы `x = torch.tensor(...)`. Иначе при скриптовании прилетит: *Unknown builtin op: aten::Tensor. Here are some suggestions: aten::tensor*. Вроде бы, даже объясняют, в чем  проблема, и понятно, что надо делать. Впрочем, понятно, если уже знаешь правильный ответ.

* Код скриптуется в контексте того модуля, где вызван `torch.jit.script`. Поэтому если где-то в недрах скриптуемого класса или функции используется, например, `math.pow`, придется в компилирующий модуль добавить `import math`. А лучше скриптовать класс там же, где он объявлен: или с помощью декоратора `@torch.jit.script`, или объявив рядом с ним дополнительную функцию, делающую из него ScriptModule. Иначе получим сообщение об ошибке *undefined value math* при попытке скомпилировать класс из модуля, в котором, казалось бы, сделан импорт `math`.

* Если где-то у вас есть конструкция вида `my_tensor[my_tensor < 10] = 0` или подобная, то при скриптовании вы получите загадочную ошибку:
```
*aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> (Tensor(a!)):*  
*Expected a value of type 'Tensor' for argument 'values' but instead found type 'int'.*  
*aten::index_put_(Tensor(a!) self, Tensor[] indices, Tensor values, bool accumulate=False) -> (Tensor(a!)):*  
*Expected a value of type 'List[Tensor]' for argument 'indices' but instead found type 'List[Optional[Tensor]]'.*  
```
Что вам нужно - это заменить число на тензор: `my_tensor[my_tensor < 10] = torch.tensor(0.).to(my_tensor.device)`. Причем не забудьте а) про соответствие типов `my_tensor` и создаваемого тензора (в данном случае - float) и б) про `.to(my_tensor.device)`. Если забудете второе, все отскриптуется, но уже в процессе выполнения при работе на GPU вас ожидает огорчение, которое будет выглядеть как загадочные слова *CUDA error: an illegal memory access was encountered*, причем без указания на место возникновения ошибки!

* Не забыть, что по умолчанию `nn.Module` и, соответственно, модели из torchvision создаются в  "в режиме поезда" (вы не поверите, но оказывается, [есть такой режим](https://fooobar.com/questions/16769103/error-when-converting-pytorch-model-to-torchscript/25666033#25666033)). При этом используется Dropout и прочие трюки из train mode, которые или сломают трассировку, или приведут к неадекватным результатам при выполнении. Не забудьте вызвать `model.eval()` перед скриптованием или трассировкой.

* Для функций и обычных классов надо скриптовать тип, для nn.Module - экземпляр

* Попытка в методе скриптуемого метода обратиться к глобальной переменной

```python
cls_thresh = 0.3
class MyModule(torch.nn.Module):
    ...
    x = r < cls_thresh
    ...
```
приведет при скриптовании к ошибке вида  *python value of type 'float' cannot be used as a value*. Надо сделать переменную атрибутом в конструкторе:
```python
cls_thresh = 0.3
class MyModule(torch.nn.Module):
    def __init__(self):
        ...
        self.cls_thresh = cls_thresh
        ...
        x = r < self.cls_thresh
        ...
```

* Еще одна тонкость возникает, если атрибут класса используется в качестве параметра среза:

```python
class FPN(nn.Module):
    def __init__(self, block, num_blocks, num_layers =5):
        ...
        self.num_layers = num_layers
    def forward(self, x):
        ...
        return (p3, p4, p5, p6, p7)[:self.num_layers]
```
приводит при скриптовании к ошибке *tuple slice indices must be integer constants*. Надо указать, что атрибут num_layers - константа и меняться не будет:
```python
class FPN(nn.Module):
    num_layers: torch.jit.Final[int]
    def __init__(self, block, num_blocks, num_layers =5):
...        
```

* В некоторых случаях там, где раньше нормально подходил тензор, требуется в явном виде передать число:

```python
xx1 = x1.clamp(min=x1[i])
```
выдает при скриптовании ошибку *`Expected a value of type 'Optional[number]' for argument 'min' but instead found type 'Tensor'.`*. Ну, тут из сообщения об ошибке понятно что делать:
```python
xx1 = x1.clamp(min=x1[i].item())
```

Перечисленные выше проблемы возникают при трассировке. Именно из-за них просто скомпилировать готовые решения в TorchScript обычно не получается, и приходится или долго заниматься массажом исходного кода (если исходный код уместно редактировать), или использовать трассировку. Но и в трассировке есть свои нюансы:

* В трассировке не работают конструкции вида

```
tensor_a.to(tensor_b.device)
```

Устройство, на которое загружается тензор, фиксируется в момент трассировки и в процессе выполнения не меняется. Частично справиться с этой проблемой можно, если объявить тензор членом `nn.Module` с типом `Parameter`. Тогда при загрузке модели он загрузится на то устройство, которое указано в функции `torch.jit.load`.

## Эпилог

Все перечисленное, конечно, создает проблемы. Но TorchScript позволяет объединить и отправить в решение как единое целое собственно модель и питоновский код, обеспечивающий пред- и постобработку. Да и время на подготовку решения к компиляции, даже несмотря на перечисленные трудности, несравнимо меньше, чем затраты на создание решения, а здесь PyTorch дает большие преимущества, так что игра стоит свеч.

![](https://habrastorage.org/webt/v0/3m/qt/v03mqtayxdfh5be4ut3nrr0c86q.jpeg)