  .text
  .globl stpncpy
  .type stpncpy, @function

#! file-offset 0x18c060
#! rip-offset  0x14c060
#! capacity    416 bytes

# Text                         #  Line  RIP       Bytes  Opcode               
.stpncpy:                      #        0x14c060  0      OPC=<label>          
  movl %esi, %esi              #  1     0x14c060  2      OPC=movl_r32_r32     
  movl %edi, %edi              #  2     0x14c062  2      OPC=movl_r32_r32     
  movl %esi, %eax              #  3     0x14c064  2      OPC=movl_r32_r32     
  orl %edi, %eax               #  4     0x14c066  2      OPC=orl_r32_r32      
  testb $0x3, %al              #  5     0x14c068  2      OPC=testb_al_imm8    
  jne .L_14c0e0                #  6     0x14c06a  2      OPC=jne_label        
  cmpl $0x3, %edx              #  7     0x14c06c  3      OPC=cmpl_r32_imm8    
  movq %rdi, %r10              #  8     0x14c06f  3      OPC=movq_r64_r64     
  movq %rsi, %r9               #  9     0x14c072  3      OPC=movq_r64_r64     
  jbe .L_14c0e0                #  10    0x14c075  2      OPC=jbe_label        
  nop                          #  11    0x14c077  1      OPC=nop              
  nop                          #  12    0x14c078  1      OPC=nop              
  nop                          #  13    0x14c079  1      OPC=nop              
  nop                          #  14    0x14c07a  1      OPC=nop              
  nop                          #  15    0x14c07b  1      OPC=nop              
  nop                          #  16    0x14c07c  1      OPC=nop              
  nop                          #  17    0x14c07d  1      OPC=nop              
  nop                          #  18    0x14c07e  1      OPC=nop              
  nop                          #  19    0x14c07f  1      OPC=nop              
.L_14c080:                     #        0x14c080  0      OPC=<label>          
  movl %r9d, %r9d              #  20    0x14c080  3      OPC=movl_r32_r32     
  movl (%r15,%r9,1), %ecx      #  21    0x14c083  4      OPC=movl_r32_m32     
  movl %ecx, %eax              #  22    0x14c087  2      OPC=movl_r32_r32     
  leal -0x1010101(%rcx), %esi  #  23    0x14c089  6      OPC=leal_r32_m16     
  notl %eax                    #  24    0x14c08f  2      OPC=notl_r32         
  andl %esi, %eax              #  25    0x14c091  2      OPC=andl_r32_r32     
  testl $0x80808080, %eax      #  26    0x14c093  6      OPC=testl_r32_imm32  
  jne .L_14c0c0                #  27    0x14c099  2      OPC=jne_label        
  subl $0x4, %edx              #  28    0x14c09b  3      OPC=subl_r32_imm8    
  nop                          #  29    0x14c09e  1      OPC=nop              
  nop                          #  30    0x14c09f  1      OPC=nop              
  nop                          #  31    0x14c0a0  1      OPC=nop              
  movl %r10d, %r10d            #  32    0x14c0a1  3      OPC=movl_r32_r32     
  movl %ecx, (%r15,%r10,1)     #  33    0x14c0a4  4      OPC=movl_m32_r32     
  addl $0x4, %r9d              #  34    0x14c0a8  4      OPC=addl_r32_imm8    
  addl $0x4, %r10d             #  35    0x14c0ac  4      OPC=addl_r32_imm8    
  cmpl $0x3, %edx              #  36    0x14c0b0  3      OPC=cmpl_r32_imm8    
  ja .L_14c080                 #  37    0x14c0b3  2      OPC=ja_label         
  nop                          #  38    0x14c0b5  1      OPC=nop              
  nop                          #  39    0x14c0b6  1      OPC=nop              
  nop                          #  40    0x14c0b7  1      OPC=nop              
  nop                          #  41    0x14c0b8  1      OPC=nop              
  nop                          #  42    0x14c0b9  1      OPC=nop              
  nop                          #  43    0x14c0ba  1      OPC=nop              
  nop                          #  44    0x14c0bb  1      OPC=nop              
  nop                          #  45    0x14c0bc  1      OPC=nop              
  nop                          #  46    0x14c0bd  1      OPC=nop              
  nop                          #  47    0x14c0be  1      OPC=nop              
  nop                          #  48    0x14c0bf  1      OPC=nop              
  nop                          #  49    0x14c0c0  1      OPC=nop              
.L_14c0c0:                     #        0x14c0c1  0      OPC=<label>          
  movq %r10, %rdi              #  50    0x14c0c1  3      OPC=movq_r64_r64     
  movq %r9, %rsi               #  51    0x14c0c4  3      OPC=movq_r64_r64     
  nop                          #  52    0x14c0c7  1      OPC=nop              
  nop                          #  53    0x14c0c8  1      OPC=nop              
  nop                          #  54    0x14c0c9  1      OPC=nop              
  nop                          #  55    0x14c0ca  1      OPC=nop              
  nop                          #  56    0x14c0cb  1      OPC=nop              
  nop                          #  57    0x14c0cc  1      OPC=nop              
  nop                          #  58    0x14c0cd  1      OPC=nop              
  nop                          #  59    0x14c0ce  1      OPC=nop              
  nop                          #  60    0x14c0cf  1      OPC=nop              
  nop                          #  61    0x14c0d0  1      OPC=nop              
  nop                          #  62    0x14c0d1  1      OPC=nop              
  nop                          #  63    0x14c0d2  1      OPC=nop              
  nop                          #  64    0x14c0d3  1      OPC=nop              
  nop                          #  65    0x14c0d4  1      OPC=nop              
  nop                          #  66    0x14c0d5  1      OPC=nop              
  nop                          #  67    0x14c0d6  1      OPC=nop              
  nop                          #  68    0x14c0d7  1      OPC=nop              
  nop                          #  69    0x14c0d8  1      OPC=nop              
  nop                          #  70    0x14c0d9  1      OPC=nop              
  nop                          #  71    0x14c0da  1      OPC=nop              
  nop                          #  72    0x14c0db  1      OPC=nop              
  nop                          #  73    0x14c0dc  1      OPC=nop              
  nop                          #  74    0x14c0dd  1      OPC=nop              
  nop                          #  75    0x14c0de  1      OPC=nop              
  nop                          #  76    0x14c0df  1      OPC=nop              
  nop                          #  77    0x14c0e0  1      OPC=nop              
.L_14c0e0:                     #        0x14c0e1  0      OPC=<label>          
  testl %edx, %edx             #  78    0x14c0e1  2      OPC=testl_r32_r32    
  je .L_14c1c0                 #  79    0x14c0e3  6      OPC=je_label_1       
  movl %esi, %esi              #  80    0x14c0e9  2      OPC=movl_r32_r32     
  movzbl (%r15,%rsi,1), %eax   #  81    0x14c0eb  5      OPC=movzbl_r32_m8    
  leal -0x1(%rdx), %r9d        #  82    0x14c0f0  4      OPC=leal_r32_m16     
  leal 0x1(%rdi), %r8d         #  83    0x14c0f4  4      OPC=leal_r32_m16     
  testb %al, %al               #  84    0x14c0f8  2      OPC=testb_r8_r8      
  movl %edi, %edi              #  85    0x14c0fa  2      OPC=movl_r32_r32     
  movb %al, (%r15,%rdi,1)      #  86    0x14c0fc  4      OPC=movb_m8_r8       
  nop                          #  87    0x14c100  1      OPC=nop              
  je .L_14c160                 #  88    0x14c101  2      OPC=je_label         
  nop                          #  89    0x14c103  1      OPC=nop              
  nop                          #  90    0x14c104  1      OPC=nop              
  nop                          #  91    0x14c105  1      OPC=nop              
  nop                          #  92    0x14c106  1      OPC=nop              
  nop                          #  93    0x14c107  1      OPC=nop              
  nop                          #  94    0x14c108  1      OPC=nop              
  nop                          #  95    0x14c109  1      OPC=nop              
  nop                          #  96    0x14c10a  1      OPC=nop              
  nop                          #  97    0x14c10b  1      OPC=nop              
  nop                          #  98    0x14c10c  1      OPC=nop              
  nop                          #  99    0x14c10d  1      OPC=nop              
  nop                          #  100   0x14c10e  1      OPC=nop              
  nop                          #  101   0x14c10f  1      OPC=nop              
  nop                          #  102   0x14c110  1      OPC=nop              
  nop                          #  103   0x14c111  1      OPC=nop              
  nop                          #  104   0x14c112  1      OPC=nop              
  nop                          #  105   0x14c113  1      OPC=nop              
  nop                          #  106   0x14c114  1      OPC=nop              
  nop                          #  107   0x14c115  1      OPC=nop              
  nop                          #  108   0x14c116  1      OPC=nop              
  nop                          #  109   0x14c117  1      OPC=nop              
  nop                          #  110   0x14c118  1      OPC=nop              
  nop                          #  111   0x14c119  1      OPC=nop              
  nop                          #  112   0x14c11a  1      OPC=nop              
  nop                          #  113   0x14c11b  1      OPC=nop              
  nop                          #  114   0x14c11c  1      OPC=nop              
  nop                          #  115   0x14c11d  1      OPC=nop              
  nop                          #  116   0x14c11e  1      OPC=nop              
  nop                          #  117   0x14c11f  1      OPC=nop              
  nop                          #  118   0x14c120  1      OPC=nop              
.L_14c120:                     #        0x14c121  0      OPC=<label>          
  addl $0x1, %esi              #  119   0x14c121  3      OPC=addl_r32_imm8    
  testl %r9d, %r9d             #  120   0x14c124  3      OPC=testl_r32_r32    
  movq %r8, %rdi               #  121   0x14c127  3      OPC=movq_r64_r64     
  je .L_14c1c0                 #  122   0x14c12a  6      OPC=je_label_1       
  movl %esi, %esi              #  123   0x14c130  2      OPC=movl_r32_r32     
  movzbl (%r15,%rsi,1), %edx   #  124   0x14c132  5      OPC=movzbl_r32_m8    
  subl $0x1, %r9d              #  125   0x14c137  4      OPC=subl_r32_imm8    
  nop                          #  126   0x14c13b  1      OPC=nop              
  nop                          #  127   0x14c13c  1      OPC=nop              
  nop                          #  128   0x14c13d  1      OPC=nop              
  nop                          #  129   0x14c13e  1      OPC=nop              
  nop                          #  130   0x14c13f  1      OPC=nop              
  nop                          #  131   0x14c140  1      OPC=nop              
  movl %r8d, %r8d              #  132   0x14c141  3      OPC=movl_r32_r32     
  movb %dl, (%r15,%r8,1)       #  133   0x14c144  4      OPC=movb_m8_r8       
  addl $0x1, %r8d              #  134   0x14c148  4      OPC=addl_r32_imm8    
  testb %dl, %dl               #  135   0x14c14c  2      OPC=testb_r8_r8      
  jne .L_14c120                #  136   0x14c14e  2      OPC=jne_label        
  xchgw %ax, %ax               #  137   0x14c150  2      OPC=xchgw_ax_r16     
  nop                          #  138   0x14c152  1      OPC=nop              
  nop                          #  139   0x14c153  1      OPC=nop              
  nop                          #  140   0x14c154  1      OPC=nop              
  nop                          #  141   0x14c155  1      OPC=nop              
  nop                          #  142   0x14c156  1      OPC=nop              
  nop                          #  143   0x14c157  1      OPC=nop              
  nop                          #  144   0x14c158  1      OPC=nop              
  nop                          #  145   0x14c159  1      OPC=nop              
  nop                          #  146   0x14c15a  1      OPC=nop              
  nop                          #  147   0x14c15b  1      OPC=nop              
  nop                          #  148   0x14c15c  1      OPC=nop              
  nop                          #  149   0x14c15d  1      OPC=nop              
  nop                          #  150   0x14c15e  1      OPC=nop              
  nop                          #  151   0x14c15f  1      OPC=nop              
  nop                          #  152   0x14c160  1      OPC=nop              
.L_14c160:                     #        0x14c161  0      OPC=<label>          
  testl %r9d, %r9d             #  153   0x14c161  3      OPC=testl_r32_r32    
  je .L_14c1a0                 #  154   0x14c164  2      OPC=je_label         
  movl %r9d, %ecx              #  155   0x14c166  3      OPC=movl_r32_r32     
  movq %r8, %rdx               #  156   0x14c169  3      OPC=movq_r64_r64     
  nop                          #  157   0x14c16c  1      OPC=nop              
  nop                          #  158   0x14c16d  1      OPC=nop              
  nop                          #  159   0x14c16e  1      OPC=nop              
  nop                          #  160   0x14c16f  1      OPC=nop              
  nop                          #  161   0x14c170  1      OPC=nop              
  nop                          #  162   0x14c171  1      OPC=nop              
  nop                          #  163   0x14c172  1      OPC=nop              
  nop                          #  164   0x14c173  1      OPC=nop              
  nop                          #  165   0x14c174  1      OPC=nop              
  nop                          #  166   0x14c175  1      OPC=nop              
  nop                          #  167   0x14c176  1      OPC=nop              
  nop                          #  168   0x14c177  1      OPC=nop              
  nop                          #  169   0x14c178  1      OPC=nop              
  nop                          #  170   0x14c179  1      OPC=nop              
  nop                          #  171   0x14c17a  1      OPC=nop              
  nop                          #  172   0x14c17b  1      OPC=nop              
  nop                          #  173   0x14c17c  1      OPC=nop              
  nop                          #  174   0x14c17d  1      OPC=nop              
  nop                          #  175   0x14c17e  1      OPC=nop              
  nop                          #  176   0x14c17f  1      OPC=nop              
  nop                          #  177   0x14c180  1      OPC=nop              
.L_14c180:                     #        0x14c181  0      OPC=<label>          
  movl %edx, %edx              #  178   0x14c181  2      OPC=movl_r32_r32     
  movb $0x0, (%r15,%rdx,1)     #  179   0x14c183  5      OPC=movb_m8_imm8     
  addl $0x1, %edx              #  180   0x14c188  3      OPC=addl_r32_imm8    
  subl $0x1, %ecx              #  181   0x14c18b  3      OPC=subl_r32_imm8    
  jne .L_14c180                #  182   0x14c18e  2      OPC=jne_label        
  leal (%r9,%r8,1), %r8d       #  183   0x14c190  4      OPC=leal_r32_m16     
  nop                          #  184   0x14c194  1      OPC=nop              
  nop                          #  185   0x14c195  1      OPC=nop              
  nop                          #  186   0x14c196  1      OPC=nop              
  nop                          #  187   0x14c197  1      OPC=nop              
  nop                          #  188   0x14c198  1      OPC=nop              
  nop                          #  189   0x14c199  1      OPC=nop              
  nop                          #  190   0x14c19a  1      OPC=nop              
  nop                          #  191   0x14c19b  1      OPC=nop              
  nop                          #  192   0x14c19c  1      OPC=nop              
  nop                          #  193   0x14c19d  1      OPC=nop              
  nop                          #  194   0x14c19e  1      OPC=nop              
  nop                          #  195   0x14c19f  1      OPC=nop              
  nop                          #  196   0x14c1a0  1      OPC=nop              
.L_14c1a0:                     #        0x14c1a1  0      OPC=<label>          
  testq %rdi, %rdi             #  197   0x14c1a1  3      OPC=testq_r64_r64    
  je .L_14c1e0                 #  198   0x14c1a4  2      OPC=je_label         
  nop                          #  199   0x14c1a6  1      OPC=nop              
  nop                          #  200   0x14c1a7  1      OPC=nop              
  nop                          #  201   0x14c1a8  1      OPC=nop              
  nop                          #  202   0x14c1a9  1      OPC=nop              
  nop                          #  203   0x14c1aa  1      OPC=nop              
  nop                          #  204   0x14c1ab  1      OPC=nop              
  nop                          #  205   0x14c1ac  1      OPC=nop              
  nop                          #  206   0x14c1ad  1      OPC=nop              
  nop                          #  207   0x14c1ae  1      OPC=nop              
  nop                          #  208   0x14c1af  1      OPC=nop              
  nop                          #  209   0x14c1b0  1      OPC=nop              
  nop                          #  210   0x14c1b1  1      OPC=nop              
  nop                          #  211   0x14c1b2  1      OPC=nop              
  nop                          #  212   0x14c1b3  1      OPC=nop              
  nop                          #  213   0x14c1b4  1      OPC=nop              
  nop                          #  214   0x14c1b5  1      OPC=nop              
  nop                          #  215   0x14c1b6  1      OPC=nop              
  nop                          #  216   0x14c1b7  1      OPC=nop              
  nop                          #  217   0x14c1b8  1      OPC=nop              
  nop                          #  218   0x14c1b9  1      OPC=nop              
  nop                          #  219   0x14c1ba  1      OPC=nop              
  nop                          #  220   0x14c1bb  1      OPC=nop              
  nop                          #  221   0x14c1bc  1      OPC=nop              
  nop                          #  222   0x14c1bd  1      OPC=nop              
  nop                          #  223   0x14c1be  1      OPC=nop              
  nop                          #  224   0x14c1bf  1      OPC=nop              
  nop                          #  225   0x14c1c0  1      OPC=nop              
.L_14c1c0:                     #        0x14c1c1  0      OPC=<label>          
  popq %r11                    #  226   0x14c1c1  2      OPC=popq_r64_1       
  movl %edi, %eax              #  227   0x14c1c3  2      OPC=movl_r32_r32     
  andl $0xffffffe0, %r11d      #  228   0x14c1c5  7      OPC=andl_r32_imm32   
  nop                          #  229   0x14c1cc  1      OPC=nop              
  nop                          #  230   0x14c1cd  1      OPC=nop              
  nop                          #  231   0x14c1ce  1      OPC=nop              
  nop                          #  232   0x14c1cf  1      OPC=nop              
  addq %r15, %r11              #  233   0x14c1d0  3      OPC=addq_r64_r64     
  jmpq %r11                    #  234   0x14c1d3  3      OPC=jmpq_r64         
  nop                          #  235   0x14c1d6  1      OPC=nop              
  nop                          #  236   0x14c1d7  1      OPC=nop              
  nop                          #  237   0x14c1d8  1      OPC=nop              
  nop                          #  238   0x14c1d9  1      OPC=nop              
  nop                          #  239   0x14c1da  1      OPC=nop              
  nop                          #  240   0x14c1db  1      OPC=nop              
  nop                          #  241   0x14c1dc  1      OPC=nop              
  nop                          #  242   0x14c1dd  1      OPC=nop              
  nop                          #  243   0x14c1de  1      OPC=nop              
  nop                          #  244   0x14c1df  1      OPC=nop              
  nop                          #  245   0x14c1e0  1      OPC=nop              
  nop                          #  246   0x14c1e1  1      OPC=nop              
  nop                          #  247   0x14c1e2  1      OPC=nop              
  nop                          #  248   0x14c1e3  1      OPC=nop              
  nop                          #  249   0x14c1e4  1      OPC=nop              
  nop                          #  250   0x14c1e5  1      OPC=nop              
  nop                          #  251   0x14c1e6  1      OPC=nop              
  nop                          #  252   0x14c1e7  1      OPC=nop              
.L_14c1e0:                     #        0x14c1e8  0      OPC=<label>          
  movq %r8, %rdi               #  253   0x14c1e8  3      OPC=movq_r64_r64     
  jmpq .L_14c1c0               #  254   0x14c1eb  2      OPC=jmpq_label       
  nop                          #  255   0x14c1ed  1      OPC=nop              
  nop                          #  256   0x14c1ee  1      OPC=nop              
  nop                          #  257   0x14c1ef  1      OPC=nop              
  nop                          #  258   0x14c1f0  1      OPC=nop              
  nop                          #  259   0x14c1f1  1      OPC=nop              
  nop                          #  260   0x14c1f2  1      OPC=nop              
  nop                          #  261   0x14c1f3  1      OPC=nop              
  nop                          #  262   0x14c1f4  1      OPC=nop              
  nop                          #  263   0x14c1f5  1      OPC=nop              
  nop                          #  264   0x14c1f6  1      OPC=nop              
  nop                          #  265   0x14c1f7  1      OPC=nop              
  nop                          #  266   0x14c1f8  1      OPC=nop              
  nop                          #  267   0x14c1f9  1      OPC=nop              
  nop                          #  268   0x14c1fa  1      OPC=nop              
  nop                          #  269   0x14c1fb  1      OPC=nop              
  nop                          #  270   0x14c1fc  1      OPC=nop              
  nop                          #  271   0x14c1fd  1      OPC=nop              
  nop                          #  272   0x14c1fe  1      OPC=nop              
  nop                          #  273   0x14c1ff  1      OPC=nop              
  nop                          #  274   0x14c200  1      OPC=nop              
  nop                          #  275   0x14c201  1      OPC=nop              
  nop                          #  276   0x14c202  1      OPC=nop              
  nop                          #  277   0x14c203  1      OPC=nop              
  nop                          #  278   0x14c204  1      OPC=nop              
  nop                          #  279   0x14c205  1      OPC=nop              
  nop                          #  280   0x14c206  1      OPC=nop              
  nop                          #  281   0x14c207  1      OPC=nop              
                                                                              
.size stpncpy, .-stpncpy
