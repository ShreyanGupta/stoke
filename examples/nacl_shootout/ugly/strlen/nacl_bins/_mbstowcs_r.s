  .text
  .globl _mbstowcs_r
  .type _mbstowcs_r, @function

#! file-offset 0x189880
#! rip-offset  0x149880
#! capacity    384 bytes

# Text                         #  Line  RIP       Bytes  Opcode    
._mbstowcs_r:                  #        0x149880  0      OPC=0     
  pushq %r14                   #  1     0x149880  2      OPC=1861  
  movl %edi, %edi              #  2     0x149882  2      OPC=1157  
  movl %r8d, %r8d              #  3     0x149884  3      OPC=1157  
  pushq %r13                   #  4     0x149887  2      OPC=1861  
  movl %ecx, %r13d             #  5     0x149889  3      OPC=1157  
  pushq %r12                   #  6     0x14988c  2      OPC=1861  
  movl %edx, %r12d             #  7     0x14988e  3      OPC=1157  
  pushq %rbx                   #  8     0x149891  1      OPC=1861  
  movl %esi, %ebx              #  9     0x149892  2      OPC=1157  
  subl $0x28, %esp             #  10    0x149894  3      OPC=2384  
  addq %r15, %rsp              #  11    0x149897  3      OPC=72    
  testq %rbx, %rbx             #  12    0x14989a  3      OPC=2438  
  nop                          #  13    0x14989d  1      OPC=1343  
  nop                          #  14    0x14989e  1      OPC=1343  
  nop                          #  15    0x14989f  1      OPC=1343  
  movq %rdi, 0x18(%rsp)        #  16    0x1498a0  5      OPC=1138  
  movq %r8, 0x10(%rsp)         #  17    0x1498a5  5      OPC=1138  
  je .L_1499c0                 #  18    0x1498aa  6      OPC=893   
  testl %ecx, %ecx             #  19    0x1498b0  2      OPC=2436  
  je .L_1499e0                 #  20    0x1498b2  6      OPC=893   
  nop                          #  21    0x1498b8  1      OPC=1343  
  nop                          #  22    0x1498b9  1      OPC=1343  
  nop                          #  23    0x1498ba  1      OPC=1343  
  nop                          #  24    0x1498bb  1      OPC=1343  
  nop                          #  25    0x1498bc  1      OPC=1343  
  nop                          #  26    0x1498bd  1      OPC=1343  
  nop                          #  27    0x1498be  1      OPC=1343  
  nop                          #  28    0x1498bf  1      OPC=1343  
.L_1498c0:                     #        0x1498c0  0      OPC=0     
  movl $0x0, 0xc(%rsp)         #  29    0x1498c0  8      OPC=1135  
  jmpq .L_149920               #  30    0x1498c8  5      OPC=930   
  nop                          #  31    0x1498cd  1      OPC=1343  
  nop                          #  32    0x1498ce  1      OPC=1343  
  nop                          #  33    0x1498cf  1      OPC=1343  
  nop                          #  34    0x1498d0  1      OPC=1343  
  nop                          #  35    0x1498d1  1      OPC=1343  
  nop                          #  36    0x1498d2  1      OPC=1343  
  nop                          #  37    0x1498d3  1      OPC=1343  
  nop                          #  38    0x1498d4  1      OPC=1343  
  nop                          #  39    0x1498d5  1      OPC=1343  
  nop                          #  40    0x1498d6  1      OPC=1343  
  nop                          #  41    0x1498d7  1      OPC=1343  
  nop                          #  42    0x1498d8  1      OPC=1343  
  nop                          #  43    0x1498d9  1      OPC=1343  
  nop                          #  44    0x1498da  1      OPC=1343  
  nop                          #  45    0x1498db  1      OPC=1343  
  nop                          #  46    0x1498dc  1      OPC=1343  
  nop                          #  47    0x1498dd  1      OPC=1343  
  nop                          #  48    0x1498de  1      OPC=1343  
  nop                          #  49    0x1498df  1      OPC=1343  
  nop                          #  50    0x1498e0  1      OPC=1343  
  nop                          #  51    0x1498e1  1      OPC=1343  
  nop                          #  52    0x1498e2  1      OPC=1343  
  nop                          #  53    0x1498e3  1      OPC=1343  
  nop                          #  54    0x1498e4  1      OPC=1343  
.L_1498e0:                     #        0x1498e5  0      OPC=0     
  je .L_1499a0                 #  55    0x1498e5  6      OPC=893   
  addl $0x1, 0xc(%rsp)         #  56    0x1498eb  5      OPC=51    
  testq %rbx, %rbx             #  57    0x1498f0  3      OPC=2438  
  je .L_149900                 #  58    0x1498f3  6      OPC=893   
  nop                          #  59    0x1498f9  1      OPC=1343  
  nop                          #  60    0x1498fa  1      OPC=1343  
  addl $0x4, %ebx              #  61    0x1498fb  3      OPC=65    
  subl $0x1, %r13d             #  62    0x1498fe  4      OPC=2384  
  nop                          #  63    0x149902  1      OPC=1343  
  nop                          #  64    0x149903  1      OPC=1343  
  nop                          #  65    0x149904  1      OPC=1343  
  nop                          #  66    0x149905  1      OPC=1343  
  nop                          #  67    0x149906  1      OPC=1343  
  nop                          #  68    0x149907  1      OPC=1343  
  nop                          #  69    0x149908  1      OPC=1343  
  nop                          #  70    0x149909  1      OPC=1343  
  nop                          #  71    0x14990a  1      OPC=1343  
.L_149900:                     #        0x14990b  0      OPC=0     
  testl %r13d, %r13d           #  72    0x14990b  3      OPC=2436  
  je .L_1499a0                 #  73    0x14990e  6      OPC=893   
  leal (%rax,%r12,1), %r12d    #  74    0x149914  4      OPC=1066  
  nop                          #  75    0x149918  1      OPC=1343  
  nop                          #  76    0x149919  1      OPC=1343  
  nop                          #  77    0x14991a  1      OPC=1343  
  nop                          #  78    0x14991b  1      OPC=1343  
  nop                          #  79    0x14991c  1      OPC=1343  
  nop                          #  80    0x14991d  1      OPC=1343  
  nop                          #  81    0x14991e  1      OPC=1343  
  nop                          #  82    0x14991f  1      OPC=1343  
  nop                          #  83    0x149920  1      OPC=1343  
  nop                          #  84    0x149921  1      OPC=1343  
  nop                          #  85    0x149922  1      OPC=1343  
  nop                          #  86    0x149923  1      OPC=1343  
  nop                          #  87    0x149924  1      OPC=1343  
  nop                          #  88    0x149925  1      OPC=1343  
  nop                          #  89    0x149926  1      OPC=1343  
  nop                          #  90    0x149927  1      OPC=1343  
  nop                          #  91    0x149928  1      OPC=1343  
  nop                          #  92    0x149929  1      OPC=1343  
  nop                          #  93    0x14992a  1      OPC=1343  
.L_149920:                     #        0x14992b  0      OPC=0     
  movl 0xff276b9(%rip), %r14d  #  94    0x14992b  7      OPC=1156  
  nop                          #  95    0x149932  1      OPC=1343  
  nop                          #  96    0x149933  1      OPC=1343  
  nop                          #  97    0x149934  1      OPC=1343  
  nop                          #  98    0x149935  1      OPC=1343  
  nop                          #  99    0x149936  1      OPC=1343  
  nop                          #  100   0x149937  1      OPC=1343  
  nop                          #  101   0x149938  1      OPC=1343  
  nop                          #  102   0x149939  1      OPC=1343  
  nop                          #  103   0x14993a  1      OPC=1343  
  nop                          #  104   0x14993b  1      OPC=1343  
  nop                          #  105   0x14993c  1      OPC=1343  
  nop                          #  106   0x14993d  1      OPC=1343  
  nop                          #  107   0x14993e  1      OPC=1343  
  nop                          #  108   0x14993f  1      OPC=1343  
  nop                          #  109   0x149940  1      OPC=1343  
  nop                          #  110   0x149941  1      OPC=1343  
  nop                          #  111   0x149942  1      OPC=1343  
  nop                          #  112   0x149943  1      OPC=1343  
  nop                          #  113   0x149944  1      OPC=1343  
  nop                          #  114   0x149945  1      OPC=1343  
  callq .__locale_charset      #  115   0x149946  5      OPC=260   
  movl %eax, %r8d              #  116   0x14994b  3      OPC=1157  
  movq %r8, (%rsp)             #  117   0x14994e  4      OPC=1138  
  nop                          #  118   0x149952  1      OPC=1343  
  nop                          #  119   0x149953  1      OPC=1343  
  nop                          #  120   0x149954  1      OPC=1343  
  nop                          #  121   0x149955  1      OPC=1343  
  nop                          #  122   0x149956  1      OPC=1343  
  nop                          #  123   0x149957  1      OPC=1343  
  nop                          #  124   0x149958  1      OPC=1343  
  nop                          #  125   0x149959  1      OPC=1343  
  nop                          #  126   0x14995a  1      OPC=1343  
  nop                          #  127   0x14995b  1      OPC=1343  
  nop                          #  128   0x14995c  1      OPC=1343  
  nop                          #  129   0x14995d  1      OPC=1343  
  nop                          #  130   0x14995e  1      OPC=1343  
  nop                          #  131   0x14995f  1      OPC=1343  
  nop                          #  132   0x149960  1      OPC=1343  
  nop                          #  133   0x149961  1      OPC=1343  
  nop                          #  134   0x149962  1      OPC=1343  
  nop                          #  135   0x149963  1      OPC=1343  
  nop                          #  136   0x149964  1      OPC=1343  
  nop                          #  137   0x149965  1      OPC=1343  
  callq .__locale_mb_cur_max   #  138   0x149966  5      OPC=260   
  movl 0x10(%rsp), %r9d        #  139   0x14996b  5      OPC=1156  
  movl %eax, %ecx              #  140   0x149970  2      OPC=1157  
  movq (%rsp), %r8             #  141   0x149972  4      OPC=1161  
  movl %r12d, %edx             #  142   0x149976  3      OPC=1157  
  movl %ebx, %esi              #  143   0x149979  2      OPC=1157  
  movl 0x18(%rsp), %edi        #  144   0x14997b  4      OPC=1156  
  xchgw %ax, %ax               #  145   0x14997f  2      OPC=3700  
  andl $0xffffffe0, %r14d      #  146   0x149981  7      OPC=131   
  nop                          #  147   0x149988  1      OPC=1343  
  nop                          #  148   0x149989  1      OPC=1343  
  nop                          #  149   0x14998a  1      OPC=1343  
  nop                          #  150   0x14998b  1      OPC=1343  
  addq %r15, %r14              #  151   0x14998c  3      OPC=72    
  callq %r14                   #  152   0x14998f  3      OPC=258   
  cmpl $0x0, %eax              #  153   0x149992  3      OPC=470   
  jge .L_1498e0                #  154   0x149995  6      OPC=907   
  movq 0x10(%rsp), %rax        #  155   0x14999b  5      OPC=1161  
  movl $0xffffffff, 0xc(%rsp)  #  156   0x1499a0  8      OPC=1135  
  movl %eax, %eax              #  157   0x1499a8  2      OPC=1157  
  movl $0x0, (%r15,%rax,1)     #  158   0x1499aa  8      OPC=1135  
.L_1499a0:                     #        0x1499b2  0      OPC=0     
  movl 0xc(%rsp), %eax         #  159   0x1499b2  4      OPC=1156  
  addl $0x28, %esp             #  160   0x1499b6  3      OPC=65    
  addq %r15, %rsp              #  161   0x1499b9  3      OPC=72    
  popq %rbx                    #  162   0x1499bc  1      OPC=1694  
  popq %r12                    #  163   0x1499bd  2      OPC=1694  
  popq %r13                    #  164   0x1499bf  2      OPC=1694  
  popq %r14                    #  165   0x1499c1  2      OPC=1694  
  popq %r11                    #  166   0x1499c3  2      OPC=1694  
  andl $0xffffffe0, %r11d      #  167   0x1499c5  7      OPC=131   
  nop                          #  168   0x1499cc  1      OPC=1343  
  nop                          #  169   0x1499cd  1      OPC=1343  
  nop                          #  170   0x1499ce  1      OPC=1343  
  nop                          #  171   0x1499cf  1      OPC=1343  
  addq %r15, %r11              #  172   0x1499d0  3      OPC=72    
  jmpq %r11                    #  173   0x1499d3  3      OPC=928   
  nop                          #  174   0x1499d6  1      OPC=1343  
  nop                          #  175   0x1499d7  1      OPC=1343  
  nop                          #  176   0x1499d8  1      OPC=1343  
.L_1499c0:                     #        0x1499d9  0      OPC=0     
  movl $0x1, %r13d             #  177   0x1499d9  6      OPC=1154  
  jmpq .L_1498c0               #  178   0x1499df  5      OPC=930   
  nop                          #  179   0x1499e4  1      OPC=1343  
  nop                          #  180   0x1499e5  1      OPC=1343  
  nop                          #  181   0x1499e6  1      OPC=1343  
  nop                          #  182   0x1499e7  1      OPC=1343  
  nop                          #  183   0x1499e8  1      OPC=1343  
  nop                          #  184   0x1499e9  1      OPC=1343  
  nop                          #  185   0x1499ea  1      OPC=1343  
  nop                          #  186   0x1499eb  1      OPC=1343  
  nop                          #  187   0x1499ec  1      OPC=1343  
  nop                          #  188   0x1499ed  1      OPC=1343  
  nop                          #  189   0x1499ee  1      OPC=1343  
  nop                          #  190   0x1499ef  1      OPC=1343  
  nop                          #  191   0x1499f0  1      OPC=1343  
  nop                          #  192   0x1499f1  1      OPC=1343  
  nop                          #  193   0x1499f2  1      OPC=1343  
  nop                          #  194   0x1499f3  1      OPC=1343  
  nop                          #  195   0x1499f4  1      OPC=1343  
  nop                          #  196   0x1499f5  1      OPC=1343  
  nop                          #  197   0x1499f6  1      OPC=1343  
  nop                          #  198   0x1499f7  1      OPC=1343  
  nop                          #  199   0x1499f8  1      OPC=1343  
.L_1499e0:                     #        0x1499f9  0      OPC=0     
  movl $0x0, 0xc(%rsp)         #  200   0x1499f9  8      OPC=1135  
  jmpq .L_1499a0               #  201   0x149a01  5      OPC=930   
  nop                          #  202   0x149a06  1      OPC=1343  
  nop                          #  203   0x149a07  1      OPC=1343  
  nop                          #  204   0x149a08  1      OPC=1343  
  nop                          #  205   0x149a09  1      OPC=1343  
  nop                          #  206   0x149a0a  1      OPC=1343  
  nop                          #  207   0x149a0b  1      OPC=1343  
  nop                          #  208   0x149a0c  1      OPC=1343  
  nop                          #  209   0x149a0d  1      OPC=1343  
  nop                          #  210   0x149a0e  1      OPC=1343  
  nop                          #  211   0x149a0f  1      OPC=1343  
  nop                          #  212   0x149a10  1      OPC=1343  
  nop                          #  213   0x149a11  1      OPC=1343  
  nop                          #  214   0x149a12  1      OPC=1343  
  nop                          #  215   0x149a13  1      OPC=1343  
  nop                          #  216   0x149a14  1      OPC=1343  
  nop                          #  217   0x149a15  1      OPC=1343  
  nop                          #  218   0x149a16  1      OPC=1343  
  nop                          #  219   0x149a17  1      OPC=1343  
  nop                          #  220   0x149a18  1      OPC=1343  
  nop                          #  221   0x149a19  1      OPC=1343  
  nop                          #  222   0x149a1a  1      OPC=1343  
  nop                          #  223   0x149a1b  1      OPC=1343  
  nop                          #  224   0x149a1c  1      OPC=1343  
  nop                          #  225   0x149a1d  1      OPC=1343  
                                                                   
.size _mbstowcs_r, .-_mbstowcs_r
