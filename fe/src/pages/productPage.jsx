import React from 'react'
import Card from './card'
import { div } from 'three/examples/jsm/nodes/Nodes.js'

export default function ProcustPage({activepagechanger , selectedprod,similaritems, viewproducts}) {
  return (
    <div className='text-white flex justify-start gap-6 w-screen h-screen'>
        <div className=' w-[25%] h-screen border-r border-r-zinc-800 bg-stone-950'>
            <div className='text-center text-2xl font-bold tracking-[10px] py-6  bg-gradient-to-b from-black via-pink-800 to-black bg-clip-text text-transparent'>SEPHORA PRODUCTS</div>

            
            <div className='mx-12 mt-5 '>

                <div className='text-5xl pb-4'>{selectedprod?.product_name}</div>
                <div className='text-stone-500 pb-3 font-semibold '>PRODUCT INFO</div>
                <div className=' bg-stone-800 rounded-3xl px-5 py-3 border-neutral-300 border'>
                    <div className='flex justify-between items-center pb-2'>
                        <div>Product ID</div>
                        <div>{selectedprod?.product_id}</div>
                    </div>
                    <div className='flex justify-between items-center pb-2'>
                        <div>Brand</div>
                        <div className='text-yellow-600'>{selectedprod?.brand_name}</div>
                    </div>
                    <div className='flex justify-between items-center pb-2'>
                        <div>Category</div>
                        <div className='text-gold-600'>{selectedprod?.primary_category}</div>
                    </div>
                    <div className='flex justify-between items-center pb-2'>
                        <div>Rating</div>
                        <div className=''>{selectedprod?.avg_rating}⭐</div>
                        {/* {selectedprod.avg_rating ? Number(selectedprod.avg_rating).toFixed(1) : "0.0"} */}
                    </div>
                    
                     <div className='flex justify-between items-center pb-2'>
                        <div>Size</div>
                        <div className=''>{selectedprod?.variation_value}</div>
                    </div>
                     <div className='flex justify-between items-center pb-2'>
                        <div>price</div>
                        <div className='text-rose-100 text-xl font-bold'>{selectedprod?.price}$</div>
                    </div>

                    

                </div>
            </div>

            <div className='mx-12 mt-16'>
                <div className='text-center py-3 font-semibold bg-rose-700 rounded-xl mt-12 cursor-pointer hover:shadow-sm hover:shadow-rose-600 duration-300' onClick={()=>activepagechanger(2)}>Back Home</div>
            </div>

            <div>
                
            </div>
        </div>

        
    
      <div className='py-6'>
        <div className='font-bold text-xl'>Similar Products</div>
        <div className='flex flex-wrap gap-2 mt-6  w-[70vw] h-[90vh] overflow-y-auto '>
        {similaritems?.map((item,index) => (
                        <div
                            key={item.product_id}
                            className="w-[270px] h-[300px]"
                            onClick={() => viewproducts(3, index,3,item.product_id)}
                        >
                            <Card
                            productId={item.product_id}
                            productName={item.product_name}
                            category = {item.primary_category}
                            price = {item.price}
                            brand = {item.brand_name}
                            love_count = {item.love_count}
                            rating = {item.avg_rating}
                            size={item.variation_value}
                            product_id = {item.product_id}
                            />
                        </div>
                    ))}
    </div>

 

        
      </div>
    </div>
  )
}
