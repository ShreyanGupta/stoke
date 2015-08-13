  .text
  .globl pthread_once
  .type pthread_once, @function

#! file-offset 0x6d5a0
#! rip-offset  0x2d5a0
#! capacity    320 bytes

# Text                         #  Line  RIP      Bytes  Opcode              
.pthread_once:                 #        0x2d5a0  0      OPC=<label>         
  movq %rbx, -0x18(%rsp)       #  1     0x2d5a0  5      OPC=movq_m64_r64    
  movl %edi, %ebx              #  2     0x2d5a5  2      OPC=movl_r32_r32    
  movq %r13, -0x8(%rsp)        #  3     0x2d5a7  5      OPC=movq_m64_r64    
  movq %r12, -0x10(%rsp)       #  4     0x2d5ac  5      OPC=movq_m64_r64    
  subl $0x18, %esp             #  5     0x2d5b1  3      OPC=subl_r32_imm8   
  addq %r15, %rsp              #  6     0x2d5b4  3      OPC=addq_r64_r64    
  movl %ebx, %ebx              #  7     0x2d5b7  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax     #  8     0x2d5b9  4      OPC=movl_r32_m32    
  movl %esi, %r13d             #  9     0x2d5bd  3      OPC=movl_r32_r32    
  testl %eax, %eax             #  10    0x2d5c0  2      OPC=testl_r32_r32   
  je .L_2d620                  #  11    0x2d5c2  2      OPC=je_label        
  nop                          #  12    0x2d5c4  1      OPC=nop             
  nop                          #  13    0x2d5c5  1      OPC=nop             
  nop                          #  14    0x2d5c6  1      OPC=nop             
  nop                          #  15    0x2d5c7  1      OPC=nop             
  nop                          #  16    0x2d5c8  1      OPC=nop             
  nop                          #  17    0x2d5c9  1      OPC=nop             
  nop                          #  18    0x2d5ca  1      OPC=nop             
  nop                          #  19    0x2d5cb  1      OPC=nop             
  nop                          #  20    0x2d5cc  1      OPC=nop             
  nop                          #  21    0x2d5cd  1      OPC=nop             
  nop                          #  22    0x2d5ce  1      OPC=nop             
  nop                          #  23    0x2d5cf  1      OPC=nop             
  nop                          #  24    0x2d5d0  1      OPC=nop             
  nop                          #  25    0x2d5d1  1      OPC=nop             
  nop                          #  26    0x2d5d2  1      OPC=nop             
  nop                          #  27    0x2d5d3  1      OPC=nop             
  nop                          #  28    0x2d5d4  1      OPC=nop             
  nop                          #  29    0x2d5d5  1      OPC=nop             
  nop                          #  30    0x2d5d6  1      OPC=nop             
  nop                          #  31    0x2d5d7  1      OPC=nop             
  nop                          #  32    0x2d5d8  1      OPC=nop             
  nop                          #  33    0x2d5d9  1      OPC=nop             
  nop                          #  34    0x2d5da  1      OPC=nop             
  nop                          #  35    0x2d5db  1      OPC=nop             
  nop                          #  36    0x2d5dc  1      OPC=nop             
  nop                          #  37    0x2d5dd  1      OPC=nop             
  nop                          #  38    0x2d5de  1      OPC=nop             
  nop                          #  39    0x2d5df  1      OPC=nop             
.L_2d5e0:                      #        0x2d5e0  0      OPC=<label>         
  movq (%rsp), %rbx            #  40    0x2d5e0  4      OPC=movq_r64_m64    
  movq 0x8(%rsp), %r12         #  41    0x2d5e4  5      OPC=movq_r64_m64    
  xorl %eax, %eax              #  42    0x2d5e9  2      OPC=xorl_r32_r32    
  movq 0x10(%rsp), %r13        #  43    0x2d5eb  5      OPC=movq_r64_m64    
  addl $0x18, %esp             #  44    0x2d5f0  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  45    0x2d5f3  3      OPC=addq_r64_r64    
  popq %r11                    #  46    0x2d5f6  2      OPC=popq_r64_1      
  nop                          #  47    0x2d5f8  1      OPC=nop             
  nop                          #  48    0x2d5f9  1      OPC=nop             
  nop                          #  49    0x2d5fa  1      OPC=nop             
  nop                          #  50    0x2d5fb  1      OPC=nop             
  nop                          #  51    0x2d5fc  1      OPC=nop             
  nop                          #  52    0x2d5fd  1      OPC=nop             
  nop                          #  53    0x2d5fe  1      OPC=nop             
  nop                          #  54    0x2d5ff  1      OPC=nop             
  andl $0xffffffe0, %r11d      #  55    0x2d600  7      OPC=andl_r32_imm32  
  nop                          #  56    0x2d607  1      OPC=nop             
  nop                          #  57    0x2d608  1      OPC=nop             
  nop                          #  58    0x2d609  1      OPC=nop             
  nop                          #  59    0x2d60a  1      OPC=nop             
  addq %r15, %r11              #  60    0x2d60b  3      OPC=addq_r64_r64    
  jmpq %r11                    #  61    0x2d60e  3      OPC=jmpq_r64        
  nop                          #  62    0x2d611  1      OPC=nop             
  nop                          #  63    0x2d612  1      OPC=nop             
  nop                          #  64    0x2d613  1      OPC=nop             
  nop                          #  65    0x2d614  1      OPC=nop             
  nop                          #  66    0x2d615  1      OPC=nop             
  nop                          #  67    0x2d616  1      OPC=nop             
  nop                          #  68    0x2d617  1      OPC=nop             
  nop                          #  69    0x2d618  1      OPC=nop             
  nop                          #  70    0x2d619  1      OPC=nop             
  nop                          #  71    0x2d61a  1      OPC=nop             
  nop                          #  72    0x2d61b  1      OPC=nop             
  nop                          #  73    0x2d61c  1      OPC=nop             
  nop                          #  74    0x2d61d  1      OPC=nop             
  nop                          #  75    0x2d61e  1      OPC=nop             
  nop                          #  76    0x2d61f  1      OPC=nop             
  nop                          #  77    0x2d620  1      OPC=nop             
  nop                          #  78    0x2d621  1      OPC=nop             
  nop                          #  79    0x2d622  1      OPC=nop             
  nop                          #  80    0x2d623  1      OPC=nop             
  nop                          #  81    0x2d624  1      OPC=nop             
  nop                          #  82    0x2d625  1      OPC=nop             
  nop                          #  83    0x2d626  1      OPC=nop             
.L_2d620:                      #        0x2d627  0      OPC=<label>         
  leal 0x4(%rbx), %r12d        #  84    0x2d627  4      OPC=leal_r32_m16    
  movl %r12d, %edi             #  85    0x2d62b  3      OPC=movl_r32_r32    
  nop                          #  86    0x2d62e  1      OPC=nop             
  nop                          #  87    0x2d62f  1      OPC=nop             
  nop                          #  88    0x2d630  1      OPC=nop             
  nop                          #  89    0x2d631  1      OPC=nop             
  nop                          #  90    0x2d632  1      OPC=nop             
  nop                          #  91    0x2d633  1      OPC=nop             
  nop                          #  92    0x2d634  1      OPC=nop             
  nop                          #  93    0x2d635  1      OPC=nop             
  nop                          #  94    0x2d636  1      OPC=nop             
  nop                          #  95    0x2d637  1      OPC=nop             
  nop                          #  96    0x2d638  1      OPC=nop             
  nop                          #  97    0x2d639  1      OPC=nop             
  nop                          #  98    0x2d63a  1      OPC=nop             
  nop                          #  99    0x2d63b  1      OPC=nop             
  nop                          #  100   0x2d63c  1      OPC=nop             
  nop                          #  101   0x2d63d  1      OPC=nop             
  nop                          #  102   0x2d63e  1      OPC=nop             
  nop                          #  103   0x2d63f  1      OPC=nop             
  nop                          #  104   0x2d640  1      OPC=nop             
  nop                          #  105   0x2d641  1      OPC=nop             
  callq .pthread_mutex_lock    #  106   0x2d642  5      OPC=callq_label     
  movl %ebx, %ebx              #  107   0x2d647  2      OPC=movl_r32_r32    
  movl (%r15,%rbx,1), %eax     #  108   0x2d649  4      OPC=movl_r32_m32    
  testl %eax, %eax             #  109   0x2d64d  2      OPC=testl_r32_r32   
  je .L_2d6a0                  #  110   0x2d64f  2      OPC=je_label        
  nop                          #  111   0x2d651  1      OPC=nop             
  nop                          #  112   0x2d652  1      OPC=nop             
  nop                          #  113   0x2d653  1      OPC=nop             
  nop                          #  114   0x2d654  1      OPC=nop             
  nop                          #  115   0x2d655  1      OPC=nop             
  nop                          #  116   0x2d656  1      OPC=nop             
  nop                          #  117   0x2d657  1      OPC=nop             
  nop                          #  118   0x2d658  1      OPC=nop             
  nop                          #  119   0x2d659  1      OPC=nop             
  nop                          #  120   0x2d65a  1      OPC=nop             
  nop                          #  121   0x2d65b  1      OPC=nop             
  nop                          #  122   0x2d65c  1      OPC=nop             
  nop                          #  123   0x2d65d  1      OPC=nop             
  nop                          #  124   0x2d65e  1      OPC=nop             
  nop                          #  125   0x2d65f  1      OPC=nop             
  nop                          #  126   0x2d660  1      OPC=nop             
  nop                          #  127   0x2d661  1      OPC=nop             
  nop                          #  128   0x2d662  1      OPC=nop             
  nop                          #  129   0x2d663  1      OPC=nop             
  nop                          #  130   0x2d664  1      OPC=nop             
  nop                          #  131   0x2d665  1      OPC=nop             
  nop                          #  132   0x2d666  1      OPC=nop             
.L_2d660:                      #        0x2d667  0      OPC=<label>         
  movl %r12d, %edi             #  133   0x2d667  3      OPC=movl_r32_r32    
  nop                          #  134   0x2d66a  1      OPC=nop             
  nop                          #  135   0x2d66b  1      OPC=nop             
  nop                          #  136   0x2d66c  1      OPC=nop             
  nop                          #  137   0x2d66d  1      OPC=nop             
  nop                          #  138   0x2d66e  1      OPC=nop             
  nop                          #  139   0x2d66f  1      OPC=nop             
  nop                          #  140   0x2d670  1      OPC=nop             
  nop                          #  141   0x2d671  1      OPC=nop             
  nop                          #  142   0x2d672  1      OPC=nop             
  nop                          #  143   0x2d673  1      OPC=nop             
  nop                          #  144   0x2d674  1      OPC=nop             
  nop                          #  145   0x2d675  1      OPC=nop             
  nop                          #  146   0x2d676  1      OPC=nop             
  nop                          #  147   0x2d677  1      OPC=nop             
  nop                          #  148   0x2d678  1      OPC=nop             
  nop                          #  149   0x2d679  1      OPC=nop             
  nop                          #  150   0x2d67a  1      OPC=nop             
  nop                          #  151   0x2d67b  1      OPC=nop             
  nop                          #  152   0x2d67c  1      OPC=nop             
  nop                          #  153   0x2d67d  1      OPC=nop             
  nop                          #  154   0x2d67e  1      OPC=nop             
  nop                          #  155   0x2d67f  1      OPC=nop             
  nop                          #  156   0x2d680  1      OPC=nop             
  nop                          #  157   0x2d681  1      OPC=nop             
  callq .pthread_mutex_unlock  #  158   0x2d682  5      OPC=callq_label     
  jmpq .L_2d5e0                #  159   0x2d687  5      OPC=jmpq_label_1    
  nop                          #  160   0x2d68c  1      OPC=nop             
  nop                          #  161   0x2d68d  1      OPC=nop             
  nop                          #  162   0x2d68e  1      OPC=nop             
  nop                          #  163   0x2d68f  1      OPC=nop             
  nop                          #  164   0x2d690  1      OPC=nop             
  nop                          #  165   0x2d691  1      OPC=nop             
  nop                          #  166   0x2d692  1      OPC=nop             
  nop                          #  167   0x2d693  1      OPC=nop             
  nop                          #  168   0x2d694  1      OPC=nop             
  nop                          #  169   0x2d695  1      OPC=nop             
  nop                          #  170   0x2d696  1      OPC=nop             
  nop                          #  171   0x2d697  1      OPC=nop             
  nop                          #  172   0x2d698  1      OPC=nop             
  nop                          #  173   0x2d699  1      OPC=nop             
  nop                          #  174   0x2d69a  1      OPC=nop             
  nop                          #  175   0x2d69b  1      OPC=nop             
  nop                          #  176   0x2d69c  1      OPC=nop             
  nop                          #  177   0x2d69d  1      OPC=nop             
  nop                          #  178   0x2d69e  1      OPC=nop             
  nop                          #  179   0x2d69f  1      OPC=nop             
  nop                          #  180   0x2d6a0  1      OPC=nop             
  nop                          #  181   0x2d6a1  1      OPC=nop             
  nop                          #  182   0x2d6a2  1      OPC=nop             
  nop                          #  183   0x2d6a3  1      OPC=nop             
  nop                          #  184   0x2d6a4  1      OPC=nop             
  nop                          #  185   0x2d6a5  1      OPC=nop             
  nop                          #  186   0x2d6a6  1      OPC=nop             
.L_2d6a0:                      #        0x2d6a7  0      OPC=<label>         
  nop                          #  187   0x2d6a7  1      OPC=nop             
  nop                          #  188   0x2d6a8  1      OPC=nop             
  nop                          #  189   0x2d6a9  1      OPC=nop             
  nop                          #  190   0x2d6aa  1      OPC=nop             
  nop                          #  191   0x2d6ab  1      OPC=nop             
  nop                          #  192   0x2d6ac  1      OPC=nop             
  nop                          #  193   0x2d6ad  1      OPC=nop             
  nop                          #  194   0x2d6ae  1      OPC=nop             
  nop                          #  195   0x2d6af  1      OPC=nop             
  nop                          #  196   0x2d6b0  1      OPC=nop             
  nop                          #  197   0x2d6b1  1      OPC=nop             
  nop                          #  198   0x2d6b2  1      OPC=nop             
  nop                          #  199   0x2d6b3  1      OPC=nop             
  nop                          #  200   0x2d6b4  1      OPC=nop             
  nop                          #  201   0x2d6b5  1      OPC=nop             
  nop                          #  202   0x2d6b6  1      OPC=nop             
  nop                          #  203   0x2d6b7  1      OPC=nop             
  nop                          #  204   0x2d6b8  1      OPC=nop             
  nop                          #  205   0x2d6b9  1      OPC=nop             
  nop                          #  206   0x2d6ba  1      OPC=nop             
  nop                          #  207   0x2d6bb  1      OPC=nop             
  nop                          #  208   0x2d6bc  1      OPC=nop             
  andl $0xffffffe0, %r13d      #  209   0x2d6bd  7      OPC=andl_r32_imm32  
  nop                          #  210   0x2d6c4  1      OPC=nop             
  nop                          #  211   0x2d6c5  1      OPC=nop             
  nop                          #  212   0x2d6c6  1      OPC=nop             
  nop                          #  213   0x2d6c7  1      OPC=nop             
  addq %r15, %r13              #  214   0x2d6c8  3      OPC=addq_r64_r64    
  callq %r13                   #  215   0x2d6cb  3      OPC=callq_r64       
  movl %ebx, %ebx              #  216   0x2d6ce  2      OPC=movl_r32_r32    
  lock                         #  217   0x2d6d0  1      OPC=lock            
  addl $0x1, (%r15,%rbx,1)     #  218   0x2d6d1  5      OPC=addl_m32_imm8   
  nop                          #  219   0x2d6d6  1      OPC=nop             
  jmpq .L_2d660                #  220   0x2d6d7  2      OPC=jmpq_label      
  nop                          #  221   0x2d6d9  1      OPC=nop             
  nop                          #  222   0x2d6da  1      OPC=nop             
  nop                          #  223   0x2d6db  1      OPC=nop             
  nop                          #  224   0x2d6dc  1      OPC=nop             
  nop                          #  225   0x2d6dd  1      OPC=nop             
  nop                          #  226   0x2d6de  1      OPC=nop             
  nop                          #  227   0x2d6df  1      OPC=nop             
  nop                          #  228   0x2d6e0  1      OPC=nop             
  nop                          #  229   0x2d6e1  1      OPC=nop             
  nop                          #  230   0x2d6e2  1      OPC=nop             
  nop                          #  231   0x2d6e3  1      OPC=nop             
  nop                          #  232   0x2d6e4  1      OPC=nop             
  nop                          #  233   0x2d6e5  1      OPC=nop             
  nop                          #  234   0x2d6e6  1      OPC=nop             
  nop                          #  235   0x2d6e7  1      OPC=nop             
  nop                          #  236   0x2d6e8  1      OPC=nop             
  nop                          #  237   0x2d6e9  1      OPC=nop             
  nop                          #  238   0x2d6ea  1      OPC=nop             
  nop                          #  239   0x2d6eb  1      OPC=nop             
  nop                          #  240   0x2d6ec  1      OPC=nop             
  nop                          #  241   0x2d6ed  1      OPC=nop             
  nop                          #  242   0x2d6ee  1      OPC=nop             
                                                                            
.size pthread_once, .-pthread_once

