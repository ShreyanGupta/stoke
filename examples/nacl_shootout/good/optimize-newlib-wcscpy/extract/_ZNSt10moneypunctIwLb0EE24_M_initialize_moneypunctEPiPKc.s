  .text
  .globl _ZNSt10moneypunctIwLb0EE24_M_initialize_moneypunctEPiPKc
  .type _ZNSt10moneypunctIwLb0EE24_M_initialize_moneypunctEPiPKc, @function

#! file-offset 0x11ba40
#! rip-offset  0xdba40
#! capacity    640 bytes

# Text                                                      #  Line  RIP      Bytes  Opcode              
._ZNSt10moneypunctIwLb0EE24_M_initialize_moneypunctEPiPKc:  #        0xdba40  0      OPC=<label>         
  pushq %rbx                                                #  1     0xdba40  1      OPC=pushq_r64_1     
  movl %edi, %ebx                                           #  2     0xdba41  2      OPC=movl_r32_r32    
  movl %ebx, %ebx                                           #  3     0xdba43  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %eax                               #  4     0xdba45  5      OPC=movl_r32_m32    
  testq %rax, %rax                                          #  5     0xdba4a  3      OPC=testq_r64_r64   
  movq %rax, %rdx                                           #  6     0xdba4d  3      OPC=movq_r64_r64    
  je .L_dbb80                                               #  7     0xdba50  6      OPC=je_label_1      
  nop                                                       #  8     0xdba56  1      OPC=nop             
  nop                                                       #  9     0xdba57  1      OPC=nop             
  nop                                                       #  10    0xdba58  1      OPC=nop             
  nop                                                       #  11    0xdba59  1      OPC=nop             
  nop                                                       #  12    0xdba5a  1      OPC=nop             
  nop                                                       #  13    0xdba5b  1      OPC=nop             
  nop                                                       #  14    0xdba5c  1      OPC=nop             
  nop                                                       #  15    0xdba5d  1      OPC=nop             
  nop                                                       #  16    0xdba5e  1      OPC=nop             
  nop                                                       #  17    0xdba5f  1      OPC=nop             
.L_dba60:                                                   #        0xdba60  0      OPC=<label>         
  movl %edx, %edx                                           #  18    0xdba60  2      OPC=movl_r32_r32    
  movl $0x2e, 0x14(%r15,%rdx,1)                             #  19    0xdba62  9      OPC=movl_m32_imm32  
  movl 0xff62063(%rip), %edx                                #  20    0xdba6b  6      OPC=movl_r32_m32    
  movl %eax, %eax                                           #  21    0xdba71  2      OPC=movl_r32_r32    
  movl $0x2c, 0x18(%r15,%rax,1)                             #  22    0xdba73  9      OPC=movl_m32_imm32  
  nop                                                       #  23    0xdba7c  1      OPC=nop             
  nop                                                       #  24    0xdba7d  1      OPC=nop             
  nop                                                       #  25    0xdba7e  1      OPC=nop             
  nop                                                       #  26    0xdba7f  1      OPC=nop             
  movl %eax, %eax                                           #  27    0xdba80  2      OPC=movl_r32_r32    
  movl $0x1003e92c, 0x8(%r15,%rax,1)                        #  28    0xdba82  9      OPC=movl_m32_imm32  
  movl %eax, %eax                                           #  29    0xdba8b  2      OPC=movl_r32_r32    
  movl $0x0, 0xc(%r15,%rax,1)                               #  30    0xdba8d  9      OPC=movl_m32_imm32  
  nop                                                       #  31    0xdba96  1      OPC=nop             
  nop                                                       #  32    0xdba97  1      OPC=nop             
  nop                                                       #  33    0xdba98  1      OPC=nop             
  nop                                                       #  34    0xdba99  1      OPC=nop             
  nop                                                       #  35    0xdba9a  1      OPC=nop             
  nop                                                       #  36    0xdba9b  1      OPC=nop             
  nop                                                       #  37    0xdba9c  1      OPC=nop             
  nop                                                       #  38    0xdba9d  1      OPC=nop             
  nop                                                       #  39    0xdba9e  1      OPC=nop             
  nop                                                       #  40    0xdba9f  1      OPC=nop             
  movl %eax, %eax                                           #  41    0xdbaa0  2      OPC=movl_r32_r32    
  movl $0x1004b250, 0x1c(%r15,%rax,1)                       #  42    0xdbaa2  9      OPC=movl_m32_imm32  
  movl %eax, %eax                                           #  43    0xdbaab  2      OPC=movl_r32_r32    
  movl %edx, 0x38(%r15,%rax,1)                              #  44    0xdbaad  5      OPC=movl_m32_r32    
  movl %eax, %eax                                           #  45    0xdbab2  2      OPC=movl_r32_r32    
  movl $0x0, 0x20(%r15,%rax,1)                              #  46    0xdbab4  9      OPC=movl_m32_imm32  
  nop                                                       #  47    0xdbabd  1      OPC=nop             
  nop                                                       #  48    0xdbabe  1      OPC=nop             
  nop                                                       #  49    0xdbabf  1      OPC=nop             
  movl %eax, %eax                                           #  50    0xdbac0  2      OPC=movl_r32_r32    
  movl $0x1004b250, 0x24(%r15,%rax,1)                       #  51    0xdbac2  9      OPC=movl_m32_imm32  
  movl %eax, %eax                                           #  52    0xdbacb  2      OPC=movl_r32_r32    
  movl $0x0, 0x28(%r15,%rax,1)                              #  53    0xdbacd  9      OPC=movl_m32_imm32  
  nop                                                       #  54    0xdbad6  1      OPC=nop             
  nop                                                       #  55    0xdbad7  1      OPC=nop             
  nop                                                       #  56    0xdbad8  1      OPC=nop             
  nop                                                       #  57    0xdbad9  1      OPC=nop             
  nop                                                       #  58    0xdbada  1      OPC=nop             
  nop                                                       #  59    0xdbadb  1      OPC=nop             
  nop                                                       #  60    0xdbadc  1      OPC=nop             
  nop                                                       #  61    0xdbadd  1      OPC=nop             
  nop                                                       #  62    0xdbade  1      OPC=nop             
  nop                                                       #  63    0xdbadf  1      OPC=nop             
  movl %eax, %eax                                           #  64    0xdbae0  2      OPC=movl_r32_r32    
  movl $0x1004b250, 0x2c(%r15,%rax,1)                       #  65    0xdbae2  9      OPC=movl_m32_imm32  
  movl %eax, %eax                                           #  66    0xdbaeb  2      OPC=movl_r32_r32    
  movl $0x0, 0x30(%r15,%rax,1)                              #  67    0xdbaed  9      OPC=movl_m32_imm32  
  nop                                                       #  68    0xdbaf6  1      OPC=nop             
  nop                                                       #  69    0xdbaf7  1      OPC=nop             
  nop                                                       #  70    0xdbaf8  1      OPC=nop             
  nop                                                       #  71    0xdbaf9  1      OPC=nop             
  nop                                                       #  72    0xdbafa  1      OPC=nop             
  nop                                                       #  73    0xdbafb  1      OPC=nop             
  nop                                                       #  74    0xdbafc  1      OPC=nop             
  nop                                                       #  75    0xdbafd  1      OPC=nop             
  nop                                                       #  76    0xdbafe  1      OPC=nop             
  nop                                                       #  77    0xdbaff  1      OPC=nop             
  movl %eax, %eax                                           #  78    0xdbb00  2      OPC=movl_r32_r32    
  movl $0x0, 0x34(%r15,%rax,1)                              #  79    0xdbb02  9      OPC=movl_m32_imm32  
  movl %ebx, %ebx                                           #  80    0xdbb0b  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %eax                               #  81    0xdbb0d  5      OPC=movl_r32_m32    
  movl 0xff94e00(%rip), %ecx                                #  82    0xdbb12  6      OPC=movl_r32_m32    
  movl %eax, %eax                                           #  83    0xdbb18  2      OPC=movl_r32_r32    
  movl %edx, 0x3c(%r15,%rax,1)                              #  84    0xdbb1a  5      OPC=movl_m32_r32    
  nop                                                       #  85    0xdbb1f  1      OPC=nop             
  movl %ebx, %ebx                                           #  86    0xdbb20  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %edx                               #  87    0xdbb22  5      OPC=movl_r32_m32    
  xorl %eax, %eax                                           #  88    0xdbb27  2      OPC=xorl_r32_r32    
  nop                                                       #  89    0xdbb29  1      OPC=nop             
  nop                                                       #  90    0xdbb2a  1      OPC=nop             
  nop                                                       #  91    0xdbb2b  1      OPC=nop             
  nop                                                       #  92    0xdbb2c  1      OPC=nop             
  nop                                                       #  93    0xdbb2d  1      OPC=nop             
  nop                                                       #  94    0xdbb2e  1      OPC=nop             
  nop                                                       #  95    0xdbb2f  1      OPC=nop             
  nop                                                       #  96    0xdbb30  1      OPC=nop             
  nop                                                       #  97    0xdbb31  1      OPC=nop             
  nop                                                       #  98    0xdbb32  1      OPC=nop             
  nop                                                       #  99    0xdbb33  1      OPC=nop             
  nop                                                       #  100   0xdbb34  1      OPC=nop             
  nop                                                       #  101   0xdbb35  1      OPC=nop             
  nop                                                       #  102   0xdbb36  1      OPC=nop             
  nop                                                       #  103   0xdbb37  1      OPC=nop             
  nop                                                       #  104   0xdbb38  1      OPC=nop             
  nop                                                       #  105   0xdbb39  1      OPC=nop             
  nop                                                       #  106   0xdbb3a  1      OPC=nop             
  nop                                                       #  107   0xdbb3b  1      OPC=nop             
  nop                                                       #  108   0xdbb3c  1      OPC=nop             
  nop                                                       #  109   0xdbb3d  1      OPC=nop             
  nop                                                       #  110   0xdbb3e  1      OPC=nop             
  nop                                                       #  111   0xdbb3f  1      OPC=nop             
.L_dbb40:                                                   #        0xdbb40  0      OPC=<label>         
  movl %ecx, %esi                                           #  112   0xdbb40  2      OPC=movl_r32_r32    
  addl $0x1, %eax                                           #  113   0xdbb42  3      OPC=addl_r32_imm8   
  movl %edx, %ebx                                           #  114   0xdbb45  2      OPC=movl_r32_r32    
  movl %esi, %esi                                           #  115   0xdbb47  2      OPC=movl_r32_r32    
  movsbl (%r15,%rsi,1), %esi                                #  116   0xdbb49  5      OPC=movsbl_r32_m8   
  addl $0x1, %ecx                                           #  117   0xdbb4e  3      OPC=addl_r32_imm8   
  addl $0x4, %edx                                           #  118   0xdbb51  3      OPC=addl_r32_imm8   
  cmpl $0xb, %eax                                           #  119   0xdbb54  3      OPC=cmpl_r32_imm8   
  movl %ebx, %ebx                                           #  120   0xdbb57  2      OPC=movl_r32_r32    
  movl %esi, 0x40(%r15,%rbx,1)                              #  121   0xdbb59  5      OPC=movl_m32_r32    
  xchgw %ax, %ax                                            #  122   0xdbb5e  2      OPC=xchgw_ax_r16    
  jne .L_dbb40                                              #  123   0xdbb60  2      OPC=jne_label       
  popq %rbx                                                 #  124   0xdbb62  1      OPC=popq_r64_1      
  popq %r11                                                 #  125   0xdbb63  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                   #  126   0xdbb65  7      OPC=andl_r32_imm32  
  nop                                                       #  127   0xdbb6c  1      OPC=nop             
  nop                                                       #  128   0xdbb6d  1      OPC=nop             
  nop                                                       #  129   0xdbb6e  1      OPC=nop             
  nop                                                       #  130   0xdbb6f  1      OPC=nop             
  addq %r15, %r11                                           #  131   0xdbb70  3      OPC=addq_r64_r64    
  jmpq %r11                                                 #  132   0xdbb73  3      OPC=jmpq_r64        
  xchgw %ax, %ax                                            #  133   0xdbb76  2      OPC=xchgw_ax_r16    
  nop                                                       #  134   0xdbb78  1      OPC=nop             
  nop                                                       #  135   0xdbb79  1      OPC=nop             
  nop                                                       #  136   0xdbb7a  1      OPC=nop             
  nop                                                       #  137   0xdbb7b  1      OPC=nop             
  nop                                                       #  138   0xdbb7c  1      OPC=nop             
  nop                                                       #  139   0xdbb7d  1      OPC=nop             
  nop                                                       #  140   0xdbb7e  1      OPC=nop             
  nop                                                       #  141   0xdbb7f  1      OPC=nop             
  nop                                                       #  142   0xdbb80  1      OPC=nop             
  nop                                                       #  143   0xdbb81  1      OPC=nop             
  nop                                                       #  144   0xdbb82  1      OPC=nop             
  nop                                                       #  145   0xdbb83  1      OPC=nop             
  nop                                                       #  146   0xdbb84  1      OPC=nop             
  nop                                                       #  147   0xdbb85  1      OPC=nop             
  nop                                                       #  148   0xdbb86  1      OPC=nop             
.L_dbb80:                                                   #        0xdbb87  0      OPC=<label>         
  movl $0x70, %edi                                          #  149   0xdbb87  5      OPC=movl_r32_imm32  
  nop                                                       #  150   0xdbb8c  1      OPC=nop             
  nop                                                       #  151   0xdbb8d  1      OPC=nop             
  nop                                                       #  152   0xdbb8e  1      OPC=nop             
  nop                                                       #  153   0xdbb8f  1      OPC=nop             
  nop                                                       #  154   0xdbb90  1      OPC=nop             
  nop                                                       #  155   0xdbb91  1      OPC=nop             
  nop                                                       #  156   0xdbb92  1      OPC=nop             
  nop                                                       #  157   0xdbb93  1      OPC=nop             
  nop                                                       #  158   0xdbb94  1      OPC=nop             
  nop                                                       #  159   0xdbb95  1      OPC=nop             
  nop                                                       #  160   0xdbb96  1      OPC=nop             
  nop                                                       #  161   0xdbb97  1      OPC=nop             
  nop                                                       #  162   0xdbb98  1      OPC=nop             
  nop                                                       #  163   0xdbb99  1      OPC=nop             
  nop                                                       #  164   0xdbb9a  1      OPC=nop             
  nop                                                       #  165   0xdbb9b  1      OPC=nop             
  nop                                                       #  166   0xdbb9c  1      OPC=nop             
  nop                                                       #  167   0xdbb9d  1      OPC=nop             
  nop                                                       #  168   0xdbb9e  1      OPC=nop             
  nop                                                       #  169   0xdbb9f  1      OPC=nop             
  nop                                                       #  170   0xdbba0  1      OPC=nop             
  nop                                                       #  171   0xdbba1  1      OPC=nop             
  callq ._Znwj                                              #  172   0xdbba2  5      OPC=callq_label     
  movl %eax, %edx                                           #  173   0xdbba7  2      OPC=movl_r32_r32    
  movl %edx, %edx                                           #  174   0xdbba9  2      OPC=movl_r32_r32    
  movb $0x0, 0x38(%r15,%rdx,1)                              #  175   0xdbbab  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  176   0xdbbb1  2      OPC=movl_r32_r32    
  movb $0x0, 0x39(%r15,%rdx,1)                              #  177   0xdbbb3  6      OPC=movb_m8_imm8    
  movq %rdx, %rax                                           #  178   0xdbbb9  3      OPC=movq_r64_r64    
  movl %edx, %edx                                           #  179   0xdbbbc  2      OPC=movl_r32_r32    
  movb $0x0, 0x3a(%r15,%rdx,1)                              #  180   0xdbbbe  6      OPC=movb_m8_imm8    
  nop                                                       #  181   0xdbbc4  1      OPC=nop             
  nop                                                       #  182   0xdbbc5  1      OPC=nop             
  nop                                                       #  183   0xdbbc6  1      OPC=nop             
  movl %edx, %edx                                           #  184   0xdbbc7  2      OPC=movl_r32_r32    
  movb $0x0, 0x3b(%r15,%rdx,1)                              #  185   0xdbbc9  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  186   0xdbbcf  2      OPC=movl_r32_r32    
  movb $0x0, 0x3c(%r15,%rdx,1)                              #  187   0xdbbd1  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  188   0xdbbd7  2      OPC=movl_r32_r32    
  movb $0x0, 0x3d(%r15,%rdx,1)                              #  189   0xdbbd9  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  190   0xdbbdf  2      OPC=movl_r32_r32    
  movb $0x0, 0x3e(%r15,%rdx,1)                              #  191   0xdbbe1  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  192   0xdbbe7  2      OPC=movl_r32_r32    
  movb $0x0, 0x3f(%r15,%rdx,1)                              #  193   0xdbbe9  6      OPC=movb_m8_imm8    
  movl %edx, %edx                                           #  194   0xdbbef  2      OPC=movl_r32_r32    
  movl $0x0, 0x4(%r15,%rdx,1)                               #  195   0xdbbf1  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  196   0xdbbfa  2      OPC=movl_r32_r32    
  movl $0x1003be78, (%r15,%rdx,1)                           #  197   0xdbbfc  8      OPC=movl_m32_imm32  
  nop                                                       #  198   0xdbc04  1      OPC=nop             
  nop                                                       #  199   0xdbc05  1      OPC=nop             
  nop                                                       #  200   0xdbc06  1      OPC=nop             
  movl %edx, %edx                                           #  201   0xdbc07  2      OPC=movl_r32_r32    
  movl $0x0, 0x8(%r15,%rdx,1)                               #  202   0xdbc09  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  203   0xdbc12  2      OPC=movl_r32_r32    
  movl $0x0, 0xc(%r15,%rdx,1)                               #  204   0xdbc14  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  205   0xdbc1d  2      OPC=movl_r32_r32    
  movb $0x0, 0x10(%r15,%rdx,1)                              #  206   0xdbc1f  6      OPC=movb_m8_imm8    
  xchgw %ax, %ax                                            #  207   0xdbc25  2      OPC=xchgw_ax_r16    
  movl %edx, %edx                                           #  208   0xdbc27  2      OPC=movl_r32_r32    
  movl $0x0, 0x14(%r15,%rdx,1)                              #  209   0xdbc29  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  210   0xdbc32  2      OPC=movl_r32_r32    
  movl $0x0, 0x18(%r15,%rdx,1)                              #  211   0xdbc34  9      OPC=movl_m32_imm32  
  nop                                                       #  212   0xdbc3d  1      OPC=nop             
  nop                                                       #  213   0xdbc3e  1      OPC=nop             
  nop                                                       #  214   0xdbc3f  1      OPC=nop             
  nop                                                       #  215   0xdbc40  1      OPC=nop             
  nop                                                       #  216   0xdbc41  1      OPC=nop             
  nop                                                       #  217   0xdbc42  1      OPC=nop             
  nop                                                       #  218   0xdbc43  1      OPC=nop             
  nop                                                       #  219   0xdbc44  1      OPC=nop             
  nop                                                       #  220   0xdbc45  1      OPC=nop             
  nop                                                       #  221   0xdbc46  1      OPC=nop             
  movl %edx, %edx                                           #  222   0xdbc47  2      OPC=movl_r32_r32    
  movl $0x0, 0x1c(%r15,%rdx,1)                              #  223   0xdbc49  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  224   0xdbc52  2      OPC=movl_r32_r32    
  movl $0x0, 0x20(%r15,%rdx,1)                              #  225   0xdbc54  9      OPC=movl_m32_imm32  
  nop                                                       #  226   0xdbc5d  1      OPC=nop             
  nop                                                       #  227   0xdbc5e  1      OPC=nop             
  nop                                                       #  228   0xdbc5f  1      OPC=nop             
  nop                                                       #  229   0xdbc60  1      OPC=nop             
  nop                                                       #  230   0xdbc61  1      OPC=nop             
  nop                                                       #  231   0xdbc62  1      OPC=nop             
  nop                                                       #  232   0xdbc63  1      OPC=nop             
  nop                                                       #  233   0xdbc64  1      OPC=nop             
  nop                                                       #  234   0xdbc65  1      OPC=nop             
  nop                                                       #  235   0xdbc66  1      OPC=nop             
  movl %edx, %edx                                           #  236   0xdbc67  2      OPC=movl_r32_r32    
  movl $0x0, 0x24(%r15,%rdx,1)                              #  237   0xdbc69  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  238   0xdbc72  2      OPC=movl_r32_r32    
  movl $0x0, 0x28(%r15,%rdx,1)                              #  239   0xdbc74  9      OPC=movl_m32_imm32  
  nop                                                       #  240   0xdbc7d  1      OPC=nop             
  nop                                                       #  241   0xdbc7e  1      OPC=nop             
  nop                                                       #  242   0xdbc7f  1      OPC=nop             
  nop                                                       #  243   0xdbc80  1      OPC=nop             
  nop                                                       #  244   0xdbc81  1      OPC=nop             
  nop                                                       #  245   0xdbc82  1      OPC=nop             
  nop                                                       #  246   0xdbc83  1      OPC=nop             
  nop                                                       #  247   0xdbc84  1      OPC=nop             
  nop                                                       #  248   0xdbc85  1      OPC=nop             
  nop                                                       #  249   0xdbc86  1      OPC=nop             
  movl %edx, %edx                                           #  250   0xdbc87  2      OPC=movl_r32_r32    
  movl $0x0, 0x2c(%r15,%rdx,1)                              #  251   0xdbc89  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  252   0xdbc92  2      OPC=movl_r32_r32    
  movl $0x0, 0x30(%r15,%rdx,1)                              #  253   0xdbc94  9      OPC=movl_m32_imm32  
  nop                                                       #  254   0xdbc9d  1      OPC=nop             
  nop                                                       #  255   0xdbc9e  1      OPC=nop             
  nop                                                       #  256   0xdbc9f  1      OPC=nop             
  nop                                                       #  257   0xdbca0  1      OPC=nop             
  nop                                                       #  258   0xdbca1  1      OPC=nop             
  nop                                                       #  259   0xdbca2  1      OPC=nop             
  nop                                                       #  260   0xdbca3  1      OPC=nop             
  nop                                                       #  261   0xdbca4  1      OPC=nop             
  nop                                                       #  262   0xdbca5  1      OPC=nop             
  nop                                                       #  263   0xdbca6  1      OPC=nop             
  movl %edx, %edx                                           #  264   0xdbca7  2      OPC=movl_r32_r32    
  movl $0x0, 0x34(%r15,%rdx,1)                              #  265   0xdbca9  9      OPC=movl_m32_imm32  
  movl %edx, %edx                                           #  266   0xdbcb2  2      OPC=movl_r32_r32    
  movb $0x0, 0x6c(%r15,%rdx,1)                              #  267   0xdbcb4  6      OPC=movb_m8_imm8    
  movl %ebx, %ebx                                           #  268   0xdbcba  2      OPC=movl_r32_r32    
  movl %edx, 0x8(%r15,%rbx,1)                               #  269   0xdbcbc  5      OPC=movl_m32_r32    
  jmpq .L_dba60                                             #  270   0xdbcc1  5      OPC=jmpq_label_1    
  nop                                                       #  271   0xdbcc6  1      OPC=nop             
                                                                                                         
.size _ZNSt10moneypunctIwLb0EE24_M_initialize_moneypunctEPiPKc, .-_ZNSt10moneypunctIwLb0EE24_M_initialize_moneypunctEPiPKc

