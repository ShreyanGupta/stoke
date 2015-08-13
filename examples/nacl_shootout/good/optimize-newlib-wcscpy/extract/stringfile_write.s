  .text
  .globl stringfile_write
  .type stringfile_write, @function

#! file-offset 0x6bc40
#! rip-offset  0x2bc40
#! capacity    416 bytes

# Text                          #  Line  RIP      Bytes  Opcode              
.stringfile_write:              #        0x2bc40  0      OPC=<label>         
  pushq %r13                    #  1     0x2bc40  2      OPC=pushq_r64_1     
  pushq %r12                    #  2     0x2bc42  2      OPC=pushq_r64_1     
  pushq %rbx                    #  3     0x2bc44  1      OPC=pushq_r64_1     
  movl %edx, %r12d              #  4     0x2bc45  3      OPC=movl_r32_r32    
  movl %edi, %ebx               #  5     0x2bc48  2      OPC=movl_r32_r32    
  movl %esi, %esi               #  6     0x2bc4a  2      OPC=movl_r32_r32    
  movl %ebx, %ebx               #  7     0x2bc4c  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %r10d  #  8     0x2bc4e  5      OPC=movl_r32_m32    
  testl %r10d, %r10d            #  9     0x2bc53  3      OPC=testl_r32_r32   
  jne .L_2bca0                  #  10    0x2bc56  2      OPC=jne_label       
  testl %edx, %edx              #  11    0x2bc58  2      OPC=testl_r32_r32   
  jne .L_2bc80                  #  12    0x2bc5a  2      OPC=jne_label       
  nop                           #  13    0x2bc5c  1      OPC=nop             
  nop                           #  14    0x2bc5d  1      OPC=nop             
  nop                           #  15    0x2bc5e  1      OPC=nop             
  nop                           #  16    0x2bc5f  1      OPC=nop             
.L_2bc60:                       #        0x2bc60  0      OPC=<label>         
  movl $0x1, %edx               #  17    0x2bc60  5      OPC=movl_r32_imm32  
  movl $0x1, %r13d              #  18    0x2bc65  6      OPC=movl_r32_imm32  
  jmpq .L_2bcc0                 #  19    0x2bc6b  2      OPC=jmpq_label      
  nop                           #  20    0x2bc6d  1      OPC=nop             
  nop                           #  21    0x2bc6e  1      OPC=nop             
  nop                           #  22    0x2bc6f  1      OPC=nop             
  nop                           #  23    0x2bc70  1      OPC=nop             
  nop                           #  24    0x2bc71  1      OPC=nop             
  nop                           #  25    0x2bc72  1      OPC=nop             
  nop                           #  26    0x2bc73  1      OPC=nop             
  nop                           #  27    0x2bc74  1      OPC=nop             
  nop                           #  28    0x2bc75  1      OPC=nop             
  nop                           #  29    0x2bc76  1      OPC=nop             
  nop                           #  30    0x2bc77  1      OPC=nop             
  nop                           #  31    0x2bc78  1      OPC=nop             
  nop                           #  32    0x2bc79  1      OPC=nop             
  nop                           #  33    0x2bc7a  1      OPC=nop             
  nop                           #  34    0x2bc7b  1      OPC=nop             
  nop                           #  35    0x2bc7c  1      OPC=nop             
  nop                           #  36    0x2bc7d  1      OPC=nop             
  nop                           #  37    0x2bc7e  1      OPC=nop             
  nop                           #  38    0x2bc7f  1      OPC=nop             
.L_2bc80:                       #        0x2bc80  0      OPC=<label>         
  leal -0x1(%r12,%rsi,1), %eax  #  39    0x2bc80  5      OPC=leal_r32_m16    
  movl %eax, %eax               #  40    0x2bc85  2      OPC=movl_r32_r32    
  cmpb $0x0, (%r15,%rax,1)      #  41    0x2bc87  5      OPC=cmpb_m8_imm8    
  jne .L_2bc60                  #  42    0x2bc8c  2      OPC=jne_label       
  nop                           #  43    0x2bc8e  1      OPC=nop             
  nop                           #  44    0x2bc8f  1      OPC=nop             
  nop                           #  45    0x2bc90  1      OPC=nop             
  nop                           #  46    0x2bc91  1      OPC=nop             
  nop                           #  47    0x2bc92  1      OPC=nop             
  nop                           #  48    0x2bc93  1      OPC=nop             
  nop                           #  49    0x2bc94  1      OPC=nop             
  nop                           #  50    0x2bc95  1      OPC=nop             
  nop                           #  51    0x2bc96  1      OPC=nop             
  nop                           #  52    0x2bc97  1      OPC=nop             
  nop                           #  53    0x2bc98  1      OPC=nop             
  nop                           #  54    0x2bc99  1      OPC=nop             
  nop                           #  55    0x2bc9a  1      OPC=nop             
  nop                           #  56    0x2bc9b  1      OPC=nop             
  nop                           #  57    0x2bc9c  1      OPC=nop             
  nop                           #  58    0x2bc9d  1      OPC=nop             
  nop                           #  59    0x2bc9e  1      OPC=nop             
  nop                           #  60    0x2bc9f  1      OPC=nop             
.L_2bca0:                       #        0x2bca0  0      OPC=<label>         
  xorl %edx, %edx               #  61    0x2bca0  2      OPC=xorl_r32_r32    
  xorl %r13d, %r13d             #  62    0x2bca2  3      OPC=xorl_r32_r32    
  nop                           #  63    0x2bca5  1      OPC=nop             
  nop                           #  64    0x2bca6  1      OPC=nop             
  nop                           #  65    0x2bca7  1      OPC=nop             
  nop                           #  66    0x2bca8  1      OPC=nop             
  nop                           #  67    0x2bca9  1      OPC=nop             
  nop                           #  68    0x2bcaa  1      OPC=nop             
  nop                           #  69    0x2bcab  1      OPC=nop             
  nop                           #  70    0x2bcac  1      OPC=nop             
  nop                           #  71    0x2bcad  1      OPC=nop             
  nop                           #  72    0x2bcae  1      OPC=nop             
  nop                           #  73    0x2bcaf  1      OPC=nop             
  nop                           #  74    0x2bcb0  1      OPC=nop             
  nop                           #  75    0x2bcb1  1      OPC=nop             
  nop                           #  76    0x2bcb2  1      OPC=nop             
  nop                           #  77    0x2bcb3  1      OPC=nop             
  nop                           #  78    0x2bcb4  1      OPC=nop             
  nop                           #  79    0x2bcb5  1      OPC=nop             
  nop                           #  80    0x2bcb6  1      OPC=nop             
  nop                           #  81    0x2bcb7  1      OPC=nop             
  nop                           #  82    0x2bcb8  1      OPC=nop             
  nop                           #  83    0x2bcb9  1      OPC=nop             
  nop                           #  84    0x2bcba  1      OPC=nop             
  nop                           #  85    0x2bcbb  1      OPC=nop             
  nop                           #  86    0x2bcbc  1      OPC=nop             
  nop                           #  87    0x2bcbd  1      OPC=nop             
  nop                           #  88    0x2bcbe  1      OPC=nop             
  nop                           #  89    0x2bcbf  1      OPC=nop             
.L_2bcc0:                       #        0x2bcc0  0      OPC=<label>         
  movl %ebx, %ebx               #  90    0x2bcc0  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rbx,1), %edi  #  91    0x2bcc2  5      OPC=movl_r32_m32    
  movl %ebx, %ebx               #  92    0x2bcc7  2      OPC=movl_r32_r32    
  movl 0xc(%r15,%rbx,1), %eax   #  93    0x2bcc9  5      OPC=movl_r32_m32    
  leal (%r12,%rdi,1), %ecx      #  94    0x2bcce  4      OPC=leal_r32_m16    
  addl %edx, %ecx               #  95    0x2bcd2  2      OPC=addl_r32_r32    
  cmpl %eax, %ecx               #  96    0x2bcd4  2      OPC=cmpl_r32_r32    
  jbe .L_2bd00                  #  97    0x2bcd6  2      OPC=jbe_label       
  leal (%r13,%rdi,1), %ecx      #  98    0x2bcd8  5      OPC=leal_r32_m16    
  cmpl %ecx, %eax               #  99    0x2bcdd  2      OPC=cmpl_r32_r32    
  nop                           #  100   0x2bcdf  1      OPC=nop             
  je .L_2bda0                   #  101   0x2bce0  6      OPC=je_label_1      
  movl %eax, %r12d              #  102   0x2bce6  3      OPC=movl_r32_r32    
  subl %edi, %r12d              #  103   0x2bce9  3      OPC=subl_r32_r32    
  subl %edx, %r12d              #  104   0x2bcec  3      OPC=subl_r32_r32    
  xchgw %ax, %ax                #  105   0x2bcef  2      OPC=xchgw_ax_r16    
  nop                           #  106   0x2bcf1  1      OPC=nop             
  nop                           #  107   0x2bcf2  1      OPC=nop             
  nop                           #  108   0x2bcf3  1      OPC=nop             
  nop                           #  109   0x2bcf4  1      OPC=nop             
  nop                           #  110   0x2bcf5  1      OPC=nop             
  nop                           #  111   0x2bcf6  1      OPC=nop             
  nop                           #  112   0x2bcf7  1      OPC=nop             
  nop                           #  113   0x2bcf8  1      OPC=nop             
  nop                           #  114   0x2bcf9  1      OPC=nop             
  nop                           #  115   0x2bcfa  1      OPC=nop             
  nop                           #  116   0x2bcfb  1      OPC=nop             
  nop                           #  117   0x2bcfc  1      OPC=nop             
  nop                           #  118   0x2bcfd  1      OPC=nop             
  nop                           #  119   0x2bcfe  1      OPC=nop             
  nop                           #  120   0x2bcff  1      OPC=nop             
.L_2bd00:                       #        0x2bd00  0      OPC=<label>         
  movl %ebx, %ebx               #  121   0x2bd00  2      OPC=movl_r32_r32    
  addl (%r15,%rbx,1), %edi      #  122   0x2bd02  4      OPC=addl_r32_m32    
  movl %r12d, %edx              #  123   0x2bd06  3      OPC=movl_r32_r32    
  nop                           #  124   0x2bd09  1      OPC=nop             
  nop                           #  125   0x2bd0a  1      OPC=nop             
  nop                           #  126   0x2bd0b  1      OPC=nop             
  nop                           #  127   0x2bd0c  1      OPC=nop             
  nop                           #  128   0x2bd0d  1      OPC=nop             
  nop                           #  129   0x2bd0e  1      OPC=nop             
  nop                           #  130   0x2bd0f  1      OPC=nop             
  nop                           #  131   0x2bd10  1      OPC=nop             
  nop                           #  132   0x2bd11  1      OPC=nop             
  nop                           #  133   0x2bd12  1      OPC=nop             
  nop                           #  134   0x2bd13  1      OPC=nop             
  nop                           #  135   0x2bd14  1      OPC=nop             
  nop                           #  136   0x2bd15  1      OPC=nop             
  nop                           #  137   0x2bd16  1      OPC=nop             
  nop                           #  138   0x2bd17  1      OPC=nop             
  nop                           #  139   0x2bd18  1      OPC=nop             
  nop                           #  140   0x2bd19  1      OPC=nop             
  nop                           #  141   0x2bd1a  1      OPC=nop             
  callq .memcpy                 #  142   0x2bd1b  5      OPC=callq_label     
  movl %ebx, %ebx               #  143   0x2bd20  2      OPC=movl_r32_r32    
  movl 0x10(%r15,%rbx,1), %eax  #  144   0x2bd22  5      OPC=movl_r32_m32    
  addl %r12d, %eax              #  145   0x2bd27  3      OPC=addl_r32_r32    
  movl %ebx, %ebx               #  146   0x2bd2a  2      OPC=movl_r32_r32    
  movl %eax, 0x10(%r15,%rbx,1)  #  147   0x2bd2c  5      OPC=movl_m32_r32    
  movl %ebx, %ebx               #  148   0x2bd31  2      OPC=movl_r32_r32    
  cmpl 0x14(%r15,%rbx,1), %eax  #  149   0x2bd33  5      OPC=cmpl_r32_m32    
  jbe .L_2bd60                  #  150   0x2bd38  2      OPC=jbe_label       
  nop                           #  151   0x2bd3a  1      OPC=nop             
  nop                           #  152   0x2bd3b  1      OPC=nop             
  nop                           #  153   0x2bd3c  1      OPC=nop             
  nop                           #  154   0x2bd3d  1      OPC=nop             
  nop                           #  155   0x2bd3e  1      OPC=nop             
  nop                           #  156   0x2bd3f  1      OPC=nop             
  movl %ebx, %ebx               #  157   0x2bd40  2      OPC=movl_r32_r32    
  movl %eax, 0x14(%r15,%rbx,1)  #  158   0x2bd42  5      OPC=movl_m32_r32    
  testl %r13d, %r13d            #  159   0x2bd47  3      OPC=testl_r32_r32   
  je .L_2bd60                   #  160   0x2bd4a  2      OPC=je_label        
  movl %ebx, %ebx               #  161   0x2bd4c  2      OPC=movl_r32_r32    
  addl (%r15,%rbx,1), %eax      #  162   0x2bd4e  4      OPC=addl_r32_m32    
  movl %eax, %eax               #  163   0x2bd52  2      OPC=movl_r32_r32    
  movb $0x0, (%r15,%rax,1)      #  164   0x2bd54  5      OPC=movb_m8_imm8    
  nop                           #  165   0x2bd59  1      OPC=nop             
  nop                           #  166   0x2bd5a  1      OPC=nop             
  nop                           #  167   0x2bd5b  1      OPC=nop             
  nop                           #  168   0x2bd5c  1      OPC=nop             
  nop                           #  169   0x2bd5d  1      OPC=nop             
  nop                           #  170   0x2bd5e  1      OPC=nop             
  nop                           #  171   0x2bd5f  1      OPC=nop             
.L_2bd60:                       #        0x2bd60  0      OPC=<label>         
  movl %r12d, %eax              #  172   0x2bd60  3      OPC=movl_r32_r32    
  nop                           #  173   0x2bd63  1      OPC=nop             
  nop                           #  174   0x2bd64  1      OPC=nop             
  nop                           #  175   0x2bd65  1      OPC=nop             
  nop                           #  176   0x2bd66  1      OPC=nop             
  nop                           #  177   0x2bd67  1      OPC=nop             
  nop                           #  178   0x2bd68  1      OPC=nop             
  nop                           #  179   0x2bd69  1      OPC=nop             
  nop                           #  180   0x2bd6a  1      OPC=nop             
  nop                           #  181   0x2bd6b  1      OPC=nop             
  nop                           #  182   0x2bd6c  1      OPC=nop             
  nop                           #  183   0x2bd6d  1      OPC=nop             
  nop                           #  184   0x2bd6e  1      OPC=nop             
  nop                           #  185   0x2bd6f  1      OPC=nop             
  nop                           #  186   0x2bd70  1      OPC=nop             
  nop                           #  187   0x2bd71  1      OPC=nop             
  nop                           #  188   0x2bd72  1      OPC=nop             
  nop                           #  189   0x2bd73  1      OPC=nop             
  nop                           #  190   0x2bd74  1      OPC=nop             
  nop                           #  191   0x2bd75  1      OPC=nop             
  nop                           #  192   0x2bd76  1      OPC=nop             
  nop                           #  193   0x2bd77  1      OPC=nop             
  nop                           #  194   0x2bd78  1      OPC=nop             
  nop                           #  195   0x2bd79  1      OPC=nop             
  nop                           #  196   0x2bd7a  1      OPC=nop             
  nop                           #  197   0x2bd7b  1      OPC=nop             
  nop                           #  198   0x2bd7c  1      OPC=nop             
  nop                           #  199   0x2bd7d  1      OPC=nop             
  nop                           #  200   0x2bd7e  1      OPC=nop             
  nop                           #  201   0x2bd7f  1      OPC=nop             
.L_2bd80:                       #        0x2bd80  0      OPC=<label>         
  popq %rbx                     #  202   0x2bd80  1      OPC=popq_r64_1      
  popq %r12                     #  203   0x2bd81  2      OPC=popq_r64_1      
  popq %r13                     #  204   0x2bd83  2      OPC=popq_r64_1      
  popq %r11                     #  205   0x2bd85  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d       #  206   0x2bd87  7      OPC=andl_r32_imm32  
  nop                           #  207   0x2bd8e  1      OPC=nop             
  nop                           #  208   0x2bd8f  1      OPC=nop             
  nop                           #  209   0x2bd90  1      OPC=nop             
  nop                           #  210   0x2bd91  1      OPC=nop             
  addq %r15, %r11               #  211   0x2bd92  3      OPC=addq_r64_r64    
  jmpq %r11                     #  212   0x2bd95  3      OPC=jmpq_r64        
  nop                           #  213   0x2bd98  1      OPC=nop             
  nop                           #  214   0x2bd99  1      OPC=nop             
  nop                           #  215   0x2bd9a  1      OPC=nop             
  nop                           #  216   0x2bd9b  1      OPC=nop             
  nop                           #  217   0x2bd9c  1      OPC=nop             
  nop                           #  218   0x2bd9d  1      OPC=nop             
  nop                           #  219   0x2bd9e  1      OPC=nop             
  nop                           #  220   0x2bd9f  1      OPC=nop             
  nop                           #  221   0x2bda0  1      OPC=nop             
  nop                           #  222   0x2bda1  1      OPC=nop             
  nop                           #  223   0x2bda2  1      OPC=nop             
  nop                           #  224   0x2bda3  1      OPC=nop             
  nop                           #  225   0x2bda4  1      OPC=nop             
  nop                           #  226   0x2bda5  1      OPC=nop             
  nop                           #  227   0x2bda6  1      OPC=nop             
.L_2bda0:                       #        0x2bda7  0      OPC=<label>         
  nop                           #  228   0x2bda7  1      OPC=nop             
  nop                           #  229   0x2bda8  1      OPC=nop             
  nop                           #  230   0x2bda9  1      OPC=nop             
  nop                           #  231   0x2bdaa  1      OPC=nop             
  nop                           #  232   0x2bdab  1      OPC=nop             
  nop                           #  233   0x2bdac  1      OPC=nop             
  nop                           #  234   0x2bdad  1      OPC=nop             
  nop                           #  235   0x2bdae  1      OPC=nop             
  nop                           #  236   0x2bdaf  1      OPC=nop             
  nop                           #  237   0x2bdb0  1      OPC=nop             
  nop                           #  238   0x2bdb1  1      OPC=nop             
  nop                           #  239   0x2bdb2  1      OPC=nop             
  nop                           #  240   0x2bdb3  1      OPC=nop             
  nop                           #  241   0x2bdb4  1      OPC=nop             
  nop                           #  242   0x2bdb5  1      OPC=nop             
  nop                           #  243   0x2bdb6  1      OPC=nop             
  nop                           #  244   0x2bdb7  1      OPC=nop             
  nop                           #  245   0x2bdb8  1      OPC=nop             
  nop                           #  246   0x2bdb9  1      OPC=nop             
  nop                           #  247   0x2bdba  1      OPC=nop             
  nop                           #  248   0x2bdbb  1      OPC=nop             
  nop                           #  249   0x2bdbc  1      OPC=nop             
  nop                           #  250   0x2bdbd  1      OPC=nop             
  nop                           #  251   0x2bdbe  1      OPC=nop             
  nop                           #  252   0x2bdbf  1      OPC=nop             
  nop                           #  253   0x2bdc0  1      OPC=nop             
  nop                           #  254   0x2bdc1  1      OPC=nop             
  callq .__errno                #  255   0x2bdc2  5      OPC=callq_label     
  movl %eax, %eax               #  256   0x2bdc7  2      OPC=movl_r32_r32    
  movl %eax, %eax               #  257   0x2bdc9  2      OPC=movl_r32_r32    
  movl $0x1c, (%r15,%rax,1)     #  258   0x2bdcb  8      OPC=movl_m32_imm32  
  xorl %eax, %eax               #  259   0x2bdd3  2      OPC=xorl_r32_r32    
  jmpq .L_2bd80                 #  260   0x2bdd5  2      OPC=jmpq_label      
  nop                           #  261   0x2bdd7  1      OPC=nop             
  nop                           #  262   0x2bdd8  1      OPC=nop             
  nop                           #  263   0x2bdd9  1      OPC=nop             
  nop                           #  264   0x2bdda  1      OPC=nop             
  nop                           #  265   0x2bddb  1      OPC=nop             
  nop                           #  266   0x2bddc  1      OPC=nop             
  nop                           #  267   0x2bddd  1      OPC=nop             
  nop                           #  268   0x2bdde  1      OPC=nop             
  nop                           #  269   0x2bddf  1      OPC=nop             
  nop                           #  270   0x2bde0  1      OPC=nop             
  nop                           #  271   0x2bde1  1      OPC=nop             
  nop                           #  272   0x2bde2  1      OPC=nop             
  nop                           #  273   0x2bde3  1      OPC=nop             
  nop                           #  274   0x2bde4  1      OPC=nop             
  nop                           #  275   0x2bde5  1      OPC=nop             
  nop                           #  276   0x2bde6  1      OPC=nop             
                                                                             
.size stringfile_write, .-stringfile_write

