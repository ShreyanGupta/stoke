  .text
  .globl sniprintf
  .type sniprintf, @function

#! file-offset 0x18bcc0
#! rip-offset  0x14bcc0
#! capacity    512 bytes

# Text                              #  Line  RIP       Bytes  Opcode                
.sniprintf:                         #        0x14bcc0  0      OPC=<label>           
  movq %rbx, -0x20(%rsp)            #  1     0x14bcc0  5      OPC=movq_m64_r64      
  movq %r12, -0x18(%rsp)            #  2     0x14bcc5  5      OPC=movq_m64_r64      
  movl %esi, %ebx                   #  3     0x14bcca  2      OPC=movl_r32_r32      
  movq %r13, -0x10(%rsp)            #  4     0x14bccc  5      OPC=movq_m64_r64      
  movq %r14, -0x8(%rsp)             #  5     0x14bcd1  5      OPC=movq_m64_r64      
  subl $0x168, %esp                 #  6     0x14bcd6  6      OPC=subl_r32_imm32    
  addq %r15, %rsp                   #  7     0x14bcdc  3      OPC=addq_r64_r64      
  nop                               #  8     0x14bcdf  1      OPC=nop               
  leal 0x13f(%rsp), %eax            #  9     0x14bce0  7      OPC=leal_r32_m16      
  movq %rcx, 0xa8(%rsp)             #  10    0x14bce7  8      OPC=movq_m64_r64      
  movq %r8, 0xb0(%rsp)              #  11    0x14bcef  8      OPC=movq_m64_r64      
  movq %r9, 0xb8(%rsp)              #  12    0x14bcf7  8      OPC=movq_m64_r64      
  nop                               #  13    0x14bcff  1      OPC=nop               
  movl %edi, %r14d                  #  14    0x14bd00  3      OPC=movl_r32_r32      
  movl %edx, %r13d                  #  15    0x14bd03  3      OPC=movl_r32_r32      
  movl %eax, %eax                   #  16    0x14bd06  2      OPC=movl_r32_r32      
  movaps %xmm7, -0xf(%r15,%rax,1)   #  17    0x14bd08  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  18    0x14bd0e  2      OPC=movl_r32_r32      
  movaps %xmm6, -0x1f(%r15,%rax,1)  #  19    0x14bd10  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  20    0x14bd16  2      OPC=movl_r32_r32      
  movaps %xmm5, -0x2f(%r15,%rax,1)  #  21    0x14bd18  6      OPC=movaps_m128_xmm   
  xchgw %ax, %ax                    #  22    0x14bd1e  2      OPC=xchgw_ax_r16      
  movl %eax, %eax                   #  23    0x14bd20  2      OPC=movl_r32_r32      
  movaps %xmm4, -0x3f(%r15,%rax,1)  #  24    0x14bd22  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  25    0x14bd28  2      OPC=movl_r32_r32      
  movaps %xmm3, -0x4f(%r15,%rax,1)  #  26    0x14bd2a  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  27    0x14bd30  2      OPC=movl_r32_r32      
  movaps %xmm2, -0x5f(%r15,%rax,1)  #  28    0x14bd32  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  29    0x14bd38  2      OPC=movl_r32_r32      
  movaps %xmm1, -0x6f(%r15,%rax,1)  #  30    0x14bd3a  6      OPC=movaps_m128_xmm   
  movl %eax, %eax                   #  31    0x14bd40  2      OPC=movl_r32_r32      
  movaps %xmm0, -0x7f(%r15,%rax,1)  #  32    0x14bd42  6      OPC=movaps_m128_xmm   
  nop                               #  33    0x14bd48  1      OPC=nop               
  nop                               #  34    0x14bd49  1      OPC=nop               
  nop                               #  35    0x14bd4a  1      OPC=nop               
  nop                               #  36    0x14bd4b  1      OPC=nop               
  nop                               #  37    0x14bd4c  1      OPC=nop               
  nop                               #  38    0x14bd4d  1      OPC=nop               
  nop                               #  39    0x14bd4e  1      OPC=nop               
  nop                               #  40    0x14bd4f  1      OPC=nop               
  nop                               #  41    0x14bd50  1      OPC=nop               
  nop                               #  42    0x14bd51  1      OPC=nop               
  nop                               #  43    0x14bd52  1      OPC=nop               
  nop                               #  44    0x14bd53  1      OPC=nop               
  nop                               #  45    0x14bd54  1      OPC=nop               
  nop                               #  46    0x14bd55  1      OPC=nop               
  nop                               #  47    0x14bd56  1      OPC=nop               
  nop                               #  48    0x14bd57  1      OPC=nop               
  nop                               #  49    0x14bd58  1      OPC=nop               
  nop                               #  50    0x14bd59  1      OPC=nop               
  nop                               #  51    0x14bd5a  1      OPC=nop               
  callq .__nacl_read_tp             #  52    0x14bd5b  5      OPC=callq_label       
  leaq -0x480(%rax), %rax           #  53    0x14bd60  7      OPC=leaq_r64_m16      
  cmpl $0x0, %ebx                   #  54    0x14bd67  3      OPC=cmpl_r32_imm8     
  movl %eax, %eax                   #  55    0x14bd6a  2      OPC=movl_r32_r32      
  movl (%r15,%rax,1), %r12d         #  56    0x14bd6c  4      OPC=movl_r32_m32      
  jl .L_14bea0                      #  57    0x14bd70  6      OPC=jl_label_1        
  leal -0x1(%rbx), %edx             #  58    0x14bd76  3      OPC=leal_r32_m16      
  xorl %eax, %eax                   #  59    0x14bd79  2      OPC=xorl_r32_r32      
  testl %ebx, %ebx                  #  60    0x14bd7b  2      OPC=testl_r32_r32     
  nop                               #  61    0x14bd7d  1      OPC=nop               
  nop                               #  62    0x14bd7e  1      OPC=nop               
  nop                               #  63    0x14bd7f  1      OPC=nop               
  leal 0x80(%rsp), %ecx             #  64    0x14bd80  7      OPC=leal_r32_m16      
  movl %esp, %esi                   #  65    0x14bd87  2      OPC=movl_r32_r32      
  movl %r12d, %edi                  #  66    0x14bd89  3      OPC=movl_r32_r32      
  cmovnel %edx, %eax                #  67    0x14bd8c  3      OPC=cmovnel_r32_r32   
  movl %r13d, %edx                  #  68    0x14bd8f  3      OPC=movl_r32_r32      
  movw $0x208, 0xc(%rsp)            #  69    0x14bd92  7      OPC=movw_m16_imm16    
  movl %eax, 0x8(%rsp)              #  70    0x14bd99  4      OPC=movl_m32_r32      
  nop                               #  71    0x14bd9d  1      OPC=nop               
  nop                               #  72    0x14bd9e  1      OPC=nop               
  nop                               #  73    0x14bd9f  1      OPC=nop               
  movl %eax, 0x14(%rsp)             #  74    0x14bda0  4      OPC=movl_m32_r32      
  leal 0x170(%rsp), %eax            #  75    0x14bda4  7      OPC=leal_r32_m16      
  movl %r14d, (%rsp)                #  76    0x14bdab  4      OPC=movl_m32_r32      
  movl %r14d, 0x10(%rsp)            #  77    0x14bdaf  5      OPC=movl_m32_r32      
  movl %eax, 0x88(%rsp)             #  78    0x14bdb4  7      OPC=movl_m32_r32      
  nop                               #  79    0x14bdbb  1      OPC=nop               
  nop                               #  80    0x14bdbc  1      OPC=nop               
  nop                               #  81    0x14bdbd  1      OPC=nop               
  nop                               #  82    0x14bdbe  1      OPC=nop               
  nop                               #  83    0x14bdbf  1      OPC=nop               
  leal 0x90(%rsp), %eax             #  84    0x14bdc0  7      OPC=leal_r32_m16      
  movw $0xffff, 0xe(%rsp)           #  85    0x14bdc7  7      OPC=movw_m16_imm16    
  movl $0x18, 0x80(%rsp)            #  86    0x14bdce  11     OPC=movl_m32_imm32    
  nop                               #  87    0x14bdd9  1      OPC=nop               
  nop                               #  88    0x14bdda  1      OPC=nop               
  nop                               #  89    0x14bddb  1      OPC=nop               
  nop                               #  90    0x14bddc  1      OPC=nop               
  nop                               #  91    0x14bddd  1      OPC=nop               
  nop                               #  92    0x14bdde  1      OPC=nop               
  nop                               #  93    0x14bddf  1      OPC=nop               
  movl $0x30, 0x84(%rsp)            #  94    0x14bde0  11     OPC=movl_m32_imm32    
  movl %eax, 0x8c(%rsp)             #  95    0x14bdeb  7      OPC=movl_m32_r32      
  nop                               #  96    0x14bdf2  1      OPC=nop               
  nop                               #  97    0x14bdf3  1      OPC=nop               
  nop                               #  98    0x14bdf4  1      OPC=nop               
  nop                               #  99    0x14bdf5  1      OPC=nop               
  nop                               #  100   0x14bdf6  1      OPC=nop               
  nop                               #  101   0x14bdf7  1      OPC=nop               
  nop                               #  102   0x14bdf8  1      OPC=nop               
  nop                               #  103   0x14bdf9  1      OPC=nop               
  nop                               #  104   0x14bdfa  1      OPC=nop               
  callq ._svfiprintf_r              #  105   0x14bdfb  5      OPC=callq_label       
  cmpl $0xffffffff, %eax            #  106   0x14be00  6      OPC=cmpl_r32_imm32    
  nop                               #  107   0x14be06  1      OPC=nop               
  nop                               #  108   0x14be07  1      OPC=nop               
  nop                               #  109   0x14be08  1      OPC=nop               
  jl .L_14be80                      #  110   0x14be09  2      OPC=jl_label          
  nop                               #  111   0x14be0b  1      OPC=nop               
  nop                               #  112   0x14be0c  1      OPC=nop               
  nop                               #  113   0x14be0d  1      OPC=nop               
  nop                               #  114   0x14be0e  1      OPC=nop               
  nop                               #  115   0x14be0f  1      OPC=nop               
  nop                               #  116   0x14be10  1      OPC=nop               
  nop                               #  117   0x14be11  1      OPC=nop               
  nop                               #  118   0x14be12  1      OPC=nop               
  nop                               #  119   0x14be13  1      OPC=nop               
  nop                               #  120   0x14be14  1      OPC=nop               
  nop                               #  121   0x14be15  1      OPC=nop               
  nop                               #  122   0x14be16  1      OPC=nop               
  nop                               #  123   0x14be17  1      OPC=nop               
  nop                               #  124   0x14be18  1      OPC=nop               
  nop                               #  125   0x14be19  1      OPC=nop               
  nop                               #  126   0x14be1a  1      OPC=nop               
  nop                               #  127   0x14be1b  1      OPC=nop               
  nop                               #  128   0x14be1c  1      OPC=nop               
  nop                               #  129   0x14be1d  1      OPC=nop               
  nop                               #  130   0x14be1e  1      OPC=nop               
  nop                               #  131   0x14be1f  1      OPC=nop               
  nop                               #  132   0x14be20  1      OPC=nop               
  nop                               #  133   0x14be21  1      OPC=nop               
  nop                               #  134   0x14be22  1      OPC=nop               
  nop                               #  135   0x14be23  1      OPC=nop               
  nop                               #  136   0x14be24  1      OPC=nop               
  nop                               #  137   0x14be25  1      OPC=nop               
.L_14be20:                          #        0x14be26  0      OPC=<label>           
  testl %ebx, %ebx                  #  138   0x14be26  2      OPC=testl_r32_r32     
  je .L_14be40                      #  139   0x14be28  2      OPC=je_label          
  movl (%rsp), %edx                 #  140   0x14be2a  3      OPC=movl_r32_m32      
  movl %edx, %edx                   #  141   0x14be2d  2      OPC=movl_r32_r32      
  movb $0x0, (%r15,%rdx,1)          #  142   0x14be2f  5      OPC=movb_m8_imm8      
  nop                               #  143   0x14be34  1      OPC=nop               
  nop                               #  144   0x14be35  1      OPC=nop               
  nop                               #  145   0x14be36  1      OPC=nop               
  nop                               #  146   0x14be37  1      OPC=nop               
  nop                               #  147   0x14be38  1      OPC=nop               
  nop                               #  148   0x14be39  1      OPC=nop               
  nop                               #  149   0x14be3a  1      OPC=nop               
  nop                               #  150   0x14be3b  1      OPC=nop               
  nop                               #  151   0x14be3c  1      OPC=nop               
  nop                               #  152   0x14be3d  1      OPC=nop               
  nop                               #  153   0x14be3e  1      OPC=nop               
  nop                               #  154   0x14be3f  1      OPC=nop               
  nop                               #  155   0x14be40  1      OPC=nop               
  nop                               #  156   0x14be41  1      OPC=nop               
  nop                               #  157   0x14be42  1      OPC=nop               
  nop                               #  158   0x14be43  1      OPC=nop               
  nop                               #  159   0x14be44  1      OPC=nop               
  nop                               #  160   0x14be45  1      OPC=nop               
.L_14be40:                          #        0x14be46  0      OPC=<label>           
  movq 0x148(%rsp), %rbx            #  161   0x14be46  8      OPC=movq_r64_m64      
  movq 0x150(%rsp), %r12            #  162   0x14be4e  8      OPC=movq_r64_m64      
  movq 0x158(%rsp), %r13            #  163   0x14be56  8      OPC=movq_r64_m64      
  movq 0x160(%rsp), %r14            #  164   0x14be5e  8      OPC=movq_r64_m64      
  addl $0x168, %esp                 #  165   0x14be66  6      OPC=addl_r32_imm32    
  addq %r15, %rsp                   #  166   0x14be6c  3      OPC=addq_r64_r64      
  popq %r11                         #  167   0x14be6f  2      OPC=popq_r64_1        
  andl $0xffffffe0, %r11d           #  168   0x14be71  7      OPC=andl_r32_imm32    
  nop                               #  169   0x14be78  1      OPC=nop               
  nop                               #  170   0x14be79  1      OPC=nop               
  nop                               #  171   0x14be7a  1      OPC=nop               
  nop                               #  172   0x14be7b  1      OPC=nop               
  addq %r15, %r11                   #  173   0x14be7c  3      OPC=addq_r64_r64      
  jmpq %r11                         #  174   0x14be7f  3      OPC=jmpq_r64          
  nop                               #  175   0x14be82  1      OPC=nop               
  nop                               #  176   0x14be83  1      OPC=nop               
  nop                               #  177   0x14be84  1      OPC=nop               
  nop                               #  178   0x14be85  1      OPC=nop               
  nop                               #  179   0x14be86  1      OPC=nop               
  nop                               #  180   0x14be87  1      OPC=nop               
  nop                               #  181   0x14be88  1      OPC=nop               
  nop                               #  182   0x14be89  1      OPC=nop               
  nop                               #  183   0x14be8a  1      OPC=nop               
  nop                               #  184   0x14be8b  1      OPC=nop               
  nop                               #  185   0x14be8c  1      OPC=nop               
.L_14be80:                          #        0x14be8d  0      OPC=<label>           
  movl %r12d, %r12d                 #  186   0x14be8d  3      OPC=movl_r32_r32      
  movl $0x4b, (%r15,%r12,1)         #  187   0x14be90  8      OPC=movl_m32_imm32    
  jmpq .L_14be20                    #  188   0x14be98  2      OPC=jmpq_label        
  nop                               #  189   0x14be9a  1      OPC=nop               
  nop                               #  190   0x14be9b  1      OPC=nop               
  nop                               #  191   0x14be9c  1      OPC=nop               
  nop                               #  192   0x14be9d  1      OPC=nop               
  nop                               #  193   0x14be9e  1      OPC=nop               
  nop                               #  194   0x14be9f  1      OPC=nop               
  nop                               #  195   0x14bea0  1      OPC=nop               
  nop                               #  196   0x14bea1  1      OPC=nop               
  nop                               #  197   0x14bea2  1      OPC=nop               
  nop                               #  198   0x14bea3  1      OPC=nop               
  nop                               #  199   0x14bea4  1      OPC=nop               
  nop                               #  200   0x14bea5  1      OPC=nop               
  nop                               #  201   0x14bea6  1      OPC=nop               
  nop                               #  202   0x14bea7  1      OPC=nop               
  nop                               #  203   0x14bea8  1      OPC=nop               
  nop                               #  204   0x14bea9  1      OPC=nop               
  nop                               #  205   0x14beaa  1      OPC=nop               
  nop                               #  206   0x14beab  1      OPC=nop               
  nop                               #  207   0x14beac  1      OPC=nop               
.L_14bea0:                          #        0x14bead  0      OPC=<label>           
  movl %r12d, %r12d                 #  208   0x14bead  3      OPC=movl_r32_r32      
  movl $0x4b, (%r15,%r12,1)         #  209   0x14beb0  8      OPC=movl_m32_imm32    
  movl $0xffffffff, %eax            #  210   0x14beb8  6      OPC=movl_r32_imm32_1  
  jmpq .L_14be40                    #  211   0x14bebe  2      OPC=jmpq_label        
  nop                               #  212   0x14bec0  1      OPC=nop               
  nop                               #  213   0x14bec1  1      OPC=nop               
  nop                               #  214   0x14bec2  1      OPC=nop               
  nop                               #  215   0x14bec3  1      OPC=nop               
  nop                               #  216   0x14bec4  1      OPC=nop               
  nop                               #  217   0x14bec5  1      OPC=nop               
  nop                               #  218   0x14bec6  1      OPC=nop               
  nop                               #  219   0x14bec7  1      OPC=nop               
  nop                               #  220   0x14bec8  1      OPC=nop               
  nop                               #  221   0x14bec9  1      OPC=nop               
  nop                               #  222   0x14beca  1      OPC=nop               
  nop                               #  223   0x14becb  1      OPC=nop               
  nop                               #  224   0x14becc  1      OPC=nop               
  nop                               #  225   0x14becd  1      OPC=nop               
                                                                                    
.size sniprintf, .-sniprintf

